import logging
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- Конфигурация AI ---
MODEL_NAME = os.environ.get(
    'EMBEDDING_MODEL', 'paraphrase-multilingual-mpnet-base-v2')
SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', 0.75))
TOP_K = int(os.environ.get('TOP_K_RESULTS', 3))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
XLS_PATH = 'data/Статьи.xls'

logger = logging.getLogger(__name__)

# --- База Знаний ---


def _load_kb_from_xls(filepath):
    """Загружает Базу Знаний из XLS файла с использованием Pandas."""
    knowledge_base_list = []
    logger.info(f"Загрузка Базы Знаний из CSV (Pandas): {filepath}")
    try:
        df = pd.read_excel(filepath)

        # Итерируемся по строкам DataFrame
        for index, row in df.iterrows():
            question = row.iloc[0]
            answer = row.iloc[1]

            if pd.notna(question) and question and pd.notna(answer) and answer:
                knowledge_base_list.append({
                    "id": f"xls_{index}",
                    "question": str(question).strip(),
                    "answer": str(answer).strip(),
                })

        logger.info(
            f"Загружено {len(knowledge_base_list)} записей из CSV с использованием Pandas.")
        return knowledge_base_list

    except FileNotFoundError:
        logger.error(f"Файл Базы Знаний не найден: {filepath}")
        return None
    except pd.errors.EmptyDataError:  # Специфичная ошибка Pandas для пустых файлов
        logger.error(f"Ошибка при чтении CSV файла {filepath}: Файл пуст.")
        return None
    except Exception as e:
        logger.error(
            f"Ошибка при чтении CSV файла {filepath} с использованием Pandas: {e}", exc_info=True)
        return None


# --- Глобальные переменные для AI ядра ---
_embedding_model = None
_kb_embeddings = None
_kb_data_indexed = []
_is_initialized = False


def initialize_ai_core():
    """
    Загружает модель эмбеддингов и индексирует Базу Знаний.
    Должна быть вызвана один раз при старте приложения.
    Возвращает True при успехе, False при ошибке.
    """
    global _embedding_model, _kb_embeddings, _kb_data_indexed, knowledge_base, _is_initialized

    if _is_initialized:
        logger.info("AI Core уже инициализирован.")
        return True

    logger.info(f"Инициализация AI Core...")
    logger.info(f"Используемое устройство: {DEVICE}")
    logger.info(f"Загрузка модели эмбеддингов: {MODEL_NAME}...")
    try:
        _embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        logger.info("Модель эмбеддингов успешно загружена.")
    except Exception as e:
        logger.error(
            f"Не удалось загрузить модель {MODEL_NAME}: {e}", exc_info=True)
        return False

    loaded_kb = _load_kb_from_xls(XLS_PATH)
    knowledge_base = loaded_kb

    logger.info("Индексация Базы Знаний...")
    # Сохраняем тексты для эмбеддингов и всю структуру для поиска
    texts_to_embed = [item['question'] for item in knowledge_base]
    # Копируем, чтобы избежать изменения исходного списка при модификации _kb_data_indexed
    _kb_data_indexed = list(knowledge_base)

    try:
        # Получаем эмбеддинги для всех текстов в БЗ
        kb_embeddings_tensor = _embedding_model.encode(
            texts_to_embed,
            convert_to_tensor=True,
            normalize_embeddings=True,  # Важно для cosine similarity
            show_progress_bar=True,
            device=DEVICE
        )
        # Переводим на CPU и в numpy
        _kb_embeddings = kb_embeddings_tensor.cpu().numpy()
        logger.info(
            f"База Знаний проиндексирована. Получено {_kb_embeddings.shape[0]} векторов.")
        _is_initialized = True
        return True
    except Exception as e:
        logger.error(
            f"Ошибка при создании эмбеддингов для БЗ: {e}", exc_info=True)
        return False


def find_relevant_knowledge(query_text: str) -> list[dict]:
    """
    Ищет релевантные чанки в проиндексированной БЗ по текстовому запросу.

    Args:
        query_text: Текст запроса пользователя.

    Returns:
        Список словарей, где каждый словарь представляет найденный релевантный
        чанк и содержит ключи 'text', 'source', 'link', 'similarity'.
        Список отсортирован по убыванию схожести.
        Возвращает пустой список, если ничего не найдено или произошла ошибка.
    """
    global _embedding_model, _kb_embeddings, _kb_data_indexed

    if not _is_initialized or _embedding_model is None or _kb_embeddings is None or _kb_embeddings.shape[0] == 0:
        logger.warning(
            "AI Core не инициализирован или БЗ пуста. Поиск невозможен.")
        return []

    logger.debug(f"Поиск по запросу: {query_text}")
    try:
        # Получаем эмбеддинг для запроса
        query_embedding_tensor = _embedding_model.encode(
            [query_text],
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=DEVICE
        )
        query_embedding = query_embedding_tensor.cpu().numpy()

        # Вычисляем косинусное сходство
        similarities = cosine_similarity(query_embedding, _kb_embeddings)[0]

        # Находим индексы и значения схожести выше порога
        relevant_indices = np.where(similarities >= SIMILARITY_THRESHOLD)[0]

        # Создаем список результатов (индекс, схожесть)
        results_with_scores = [
            (idx, similarities[idx]) for idx in relevant_indices
        ]

        # Сортируем по схожести в убывающем порядке
        results_with_scores.sort(key=lambda item: item[1], reverse=True)

        # Формируем финальный список результатов (берем TOP_K)
        final_results = []
        for idx, similarity in results_with_scores[:TOP_K]:
            result_item = _kb_data_indexed[idx]
            final_results.append({
                "id": result_item.get('id'),
                "text": result_item['answer'],
                "similarity": float(similarity)
            })

        logger.info(
            f"Найдено {len(final_results)} релевантных результатов для запроса '{query_text}'.")
        return final_results

    except Exception as e:
        logger.error(
            f"Ошибка во время поиска по запросу '{query_text}': {e}", exc_info=True)
        return []


def get_ai_status():
    """Возвращает статус инициализации AI ядра."""
    return _is_initialized

# --- Дополнительные функции AI (можно добавить позже) ---
# def correct_spelling(text: str) -> str: ...
# def classify_intent(text: str) -> str: ...
