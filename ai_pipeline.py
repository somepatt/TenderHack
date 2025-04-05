import logging
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Any, Optional, Tuple

# --- Конфигурация AI ---
RETRIEVAL_MODEL_NAME = os.environ.get(
    'EMBEDDING_MODEL', 'paraphrase-multilingual-mpnet-base-v2')
SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', 0.75))
TOP_K = int(os.environ.get('TOP_K_RESULTS', 3))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
XLS_PATH = 'data/Статьи.xls'
CPU_DTYPE = torch.float32

GENERATION_MODEL_NAME = os.environ.get(
    'GENERATION_MODEL', 'google/gemma-2b-it')
USE_QUANTIZATION = False
MAX_NEW_TOKENS_RAG = int(os.environ.get(
    'MAX_NEW_TOKENS_RAG', 200))  # Для ответа по БЗ
MAX_NEW_TOKENS_LIVE = int(os.environ.get(
    'MAX_NEW_TOKENS_LIVE', 100))  # Для "живого" ответа


logger = logging.getLogger(__name__)

# --- Категории Запросов ---
QUERY_CATEGORIES = [
    "Вопросы о функционале портала",
    "Жалобы",
    "Запросы на техническую помощь",
    "Запросы по учётной записи",
    "Общие вопросы",  # Например, о законодательстве (44/223-ФЗ)
    "Обратная связь",  # Предложения, благодарности
    "Приветствие/Прощание",  # Добавим для простых диалогов
    "Другое"  # Если не подходит ни одна категория
]
# Категории, для которых нужно искать в БЗ
SEARCH_KB_CATEGORIES = [
    "Вопросы о функционале портала",
    "Запросы на техническую помощь",
    "Запросы по учётной записи",
    "Общие вопросы",
]

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
_tokenizer_llm = None
_kb_data_indexed = []
_is_initialized_retrieval = False
_generation_pipeline = None
_is_initialized_generation = False


def initialize_ai_core():
    """
    Загружает модель эмбеддингов и индексирует Базу Знаний.
    Должна быть вызвана один раз при старте приложения.
    Возвращает True при успехе, False при ошибке.
    """
    global _embedding_model, _kb_embeddings, _kb_data_indexed, knowledge_base, _is_initialized_retrieval
    global _generation_pipeline, _tokenizer_llm, _is_initialized_generation

    if _is_initialized_retrieval and _is_initialized_generation:
        logger.info("AI Core (Retrieval & Generation) уже инициализирован.")
        return True

    logger.info(f"Инициализация AI Core...")
    llm_device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_dtype = torch.float16 if torch.cuda.is_available() else CPU_DTYPE

    # --- 1. Retrieval ---
    if not _is_initialized_retrieval:
        logger.info(f"Инициализация Retrieval части...")
        # ... (загрузка sentence-transformer и индексация БЗ) ...
        try:
            _retrieval_model = SentenceTransformer(
                RETRIEVAL_MODEL_NAME, device=DEVICE)
            logger.info("Retrieval модель успешно загружена.")
        except Exception as e:
            logger.error(f"Не удалось загрузить Retrieval модель: {e}")
            return False
        loaded_kb = _load_kb_from_xls(XLS_PATH)
        if loaded_kb is None:
            return False
        _kb_data_indexed = loaded_kb
        if _kb_data_indexed:
            texts_to_embed = [item['question'] for item in _kb_data_indexed]
            try:
                kb_embeddings_tensor = _retrieval_model.encode(
                    texts_to_embed, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True, device=DEVICE)
                _kb_embeddings = kb_embeddings_tensor.cpu().numpy()
                logger.info(
                    f"БЗ проиндексирована для Retrieval: {_kb_embeddings.shape[0]} векторов.")
            except Exception as e:
                logger.error(f"Ошибка создания эмбеддингов БЗ: {e}")
                return False
        else:
            _kb_embeddings = np.array([])
        _is_initialized_retrieval = True
    else:
        logger.info("Retrieval часть уже инициализирована.")

    # --- 2. Generation ---
    if not _is_initialized_generation:
        logger.info(
            f"Инициализация Generation части ({GENERATION_MODEL_NAME})...")
        try:
            _tokenizer_llm = AutoTokenizer.from_pretrained(
                GENERATION_MODEL_NAME)
            model_llm = AutoModelForCausalLM.from_pretrained(
                GENERATION_MODEL_NAME, device_map=llm_device, torch_dtype=llm_dtype, trust_remote_code=True)
            _generation_pipeline = pipeline(
                "text-generation", model=model_llm, tokenizer=_tokenizer_llm)
            logger.info("Generation pipeline создан.")
            _is_initialized_generation = True
        except Exception as e:
            logger.error(
                f"Не удалось загрузить Generation модель {GENERATION_MODEL_NAME}: {e}", exc_info=True)
            _generation_pipeline = None
            # Помечаем как инициализированную (с ошибкой)
            _is_initialized_generation = True
            logger.warning("Generation модель не загружена!")
    else:
        logger.info("Generation часть уже инициализирована.")

    return _is_initialized_retrieval


def classify_query_type_with_llm(user_query: str) -> Optional[str]:
    """
    Классифицирует запрос пользователя по заданным категориям с помощью LLM.
    """
    global _generation_pipeline, _tokenizer_llm, QUERY_CATEGORIES

    if _generation_pipeline is None:
        logger.warning(
            "Generation pipeline недоступен, классификация невозможна.")
        return "Другое"  # Возвращаем категорию по умолчанию

    # Формируем промпт для классификации
    category_list_str = "\n".join([f"- {cat}" for cat in QUERY_CATEGORIES])
    prompt = f"""<start_of_turn>user
Определи наиболее подходящую категорию для следующего ЗАПРОСА ПОЛЬЗОВАТЕЛЯ из списка ниже. Ответь ТОЛЬКО названием одной категории из списка.

КАТЕГОРИИ:
{category_list_str}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{user_query}
<end_of_turn>
<start_of_turn>model
Категория: """  # Модель должна продолжить после "Категория: "

    logger.debug(f"Промпт для классификации:\n{prompt}")
    try:
        logger.info(f"Классификация запроса: '{user_query[:50]}...'")
        # Используем небольшое кол-во токенов для ответа
        generation_args = {
            "max_new_tokens": 20,  # Должно хватить для названия категории
            "temperature": 0.1,  # Низкая температура для точности
            "top_p": 0.9,
            "do_sample": True,
            "eos_token_id": _tokenizer_llm.eos_token_id,
        }
        results = _generation_pipeline(prompt, **generation_args)
        generated_text_full = results[0]['generated_text']
        raw_category = generated_text_full[len(prompt):].strip()

        # Пост-обработка: ищем точное совпадение с нашими категориями
        best_match = None
        highest_similarity = -1  # Используем простую проверку на вхождение
        cleaned_raw_category = raw_category.split(
            '\n')[0].strip().lower()  # Берем первую строку и чистим

        for category in QUERY_CATEGORIES:
            if category.lower() == cleaned_raw_category:
                best_match = category
                break  # Точное совпадение найдено
            # Если точного нет, ищем частичное
            if category.lower() in cleaned_raw_category:
                if len(category) > highest_similarity:  # Предпочитаем более длинное совпадение
                    highest_similarity = len(category)
                    best_match = category

        if best_match:
            logger.info(
                f"Запрос классифицирован как: '{best_match}' (raw: '{raw_category}')")
            return best_match
        else:
            logger.warning(
                f"Не удалось точно классифицировать запрос. Ответ LLM: '{raw_category}'. Возвращаем 'Другое'.")
            return "Другое"

    except Exception as e:
        logger.error(
            f"Ошибка классификации запроса '{user_query[:50]}...': {e}", exc_info=True)
        return "Другое"  # Возвращаем 'Другое' при ошибке


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


def generate_answer_with_llm(user_query: str, context_list: list[dict]) -> Optional[str]:
    # ... (код без изменений, использует MAX_NEW_TOKENS_RAG) ...
    global _generation_pipeline, _tokenizer_llm
    if _generation_pipeline is None or not context_list:
        return None
    # ... (формирование промпта Gemma RAG) ...
    context_str = ""
    sources = set()
    for i, ctx in enumerate(context_list):
        context_str += f"Контекст {i+1}:\n Найденный Вопрос: {ctx['question']}\n Ответ из Базы Знаний: {ctx['answer']}\n\n"
        if ctx.get('source'):
            sources.add(ctx['source'])
    source_str = ", ".join(sources) if sources else "База Знаний"
    prompt = f"""<start_of_turn>user
Ты - ИИ-ассистент Портала Поставщиков Москвы. Ответь на ВОПРОС ПОЛЬЗОВАТЕЛЯ кратко и четко, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном КОНТЕКСТЕ. Не добавляй никакой информации, которой нет в КОНТЕКСТЕ. В конце ответа укажи ИСТОЧНИК(И).

КОНТЕКСТ:
{context_str}
ИСТОЧНИК(И): {source_str}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{user_query}
<end_of_turn>
<start_of_turn>model
"""
    logger.debug(f"Промпт для LLM (Gemma RAG):\n{prompt}")
    try:
        logger.info("Генерация RAG ответа с помощью LLM (Gemma)...")
        generation_args = {"max_new_tokens": MAX_NEW_TOKENS_RAG, "temperature": 0.7, "top_p": 0.9,
                           "top_k": 50, "do_sample": True, "eos_token_id": _tokenizer_llm.eos_token_id}
        results = _generation_pipeline(prompt, **generation_args)
        generated_text_full = results[0]['generated_text']
        answer_only = generated_text_full[len(prompt):].strip()
        if answer_only.endswith("<end_of_turn>"):
            answer_only = answer_only[:-len("<end_of_turn>")].strip()
        logger.info(f"Сгенерирован RAG ответ LLM (Gemma): {answer_only}")
        return answer_only
    except Exception as e:
        logger.error(f"Ошибка генерации RAG ответа LLM (Gemma): {e}")
        return None

# --- НОВАЯ Функция для "Живого" ответа ---


def generate_live_response_with_llm(user_query: str, query_category: str) -> Optional[str]:
    """
    Генерирует ответ для категорий, не требующих поиска по БЗ (жалобы, обратная связь и т.д.).
    """
    global _generation_pipeline, _tokenizer_llm

    if _generation_pipeline is None:
        logger.warning(
            "Generation pipeline недоступен, генерация 'живого' ответа невозможна.")
        # Возвращаем заглушку или стандартный ответ
        return "Спасибо за ваше сообщение. Я передам его специалистам."

    # Формируем промпт для "живого" ответа, учитывая категорию
    system_prompt = "Ты - вежливый и эмпатичный ИИ-ассистент Портала Поставщиков Москвы."
    instruction = ""
    if query_category == "Жалобы":
        instruction = "Пользователь оставил жалобу. Вырази сожаление и сообщи, что жалоба будет рассмотрена специалистами. Предложи обратиться к оператору поддержки для срочных вопросов."
    elif query_category == "Обратная связь":
        instruction = "Пользователь оставил обратную связь (предложение или благодарность). Поблагодари пользователя за его мнение и сообщи, что оно будет учтено."
    elif query_category == "Приветствие/Прощание":
        instruction = "Пользователь поздоровался или попрощался. Ответь вежливо и кратко."
    else:  # Категория "Другое"
        instruction = "Пользователь задал вопрос или оставил сообщение, не относящееся к стандартным категориям. Ответь вежливо, но сообщи, что ты можешь помочь только с вопросами по Порталу Поставщиков или Базе Знаний. Если вопрос важный, предложи обратиться к оператору."

    prompt = f"""<start_of_turn>user
{system_prompt} {instruction}

СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:
{user_query}
<end_of_turn>
<start_of_turn>model
"""

    logger.debug(f"Промпт для LLM (Live Response):\n{prompt}")
    try:
        logger.info(
            f"Генерация 'живого' ответа для категории '{query_category}'...")
        generation_args = {
            "max_new_tokens": MAX_NEW_TOKENS_LIVE,  # Короткий ответ
            "temperature": 0.75,  # Чуть больше креативности
            "top_p": 0.9,
            "do_sample": True,
            "eos_token_id": _tokenizer_llm.eos_token_id,
        }
        results = _generation_pipeline(prompt, **generation_args)
        generated_text_full = results[0]['generated_text']
        answer_only = generated_text_full[len(prompt):].strip()
        if answer_only.endswith("<end_of_turn>"):
            answer_only = answer_only[:-len("<end_of_turn>")].strip()

        logger.info(f"Сгенерирован 'живой' ответ LLM: {answer_only}")
        return answer_only

    except Exception as e:
        logger.error(
            f"Ошибка генерации 'живого' ответа LLM: {e}", exc_info=True)
        return "Спасибо за ваше сообщение. Возникла техническая ошибка при обработке."


def get_ai_status():
    # Возвращаем статус обеих систем
    return _is_initialized_retrieval, _is_initialized_generation


def get_query_embedding(query_text: str) -> Optional[np.ndarray]:
    if not _is_initialized_retrieval or _retrieval_model is None:
        return None
    try:
        embedding_tensor = _retrieval_model.encode(
            [query_text], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
        return embedding_tensor.cpu().numpy()[0]
    except Exception as e:
        logger.error(f"Ошибка получения эмбеддинга: {e}")
        return None


def cluster_queries(embeddings: List[np.ndarray], num_clusters: int = 10):
    from sklearn.cluster import KMeans
    if not embeddings:
        return None, None
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(np.array(embeddings))
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    logger.info(f"Запросы кластеризованы на {num_clusters} тем.")
    return labels, centroids

# --- Дополнительные функции AI (можно добавить позже) ---
# def correct_spelling(text: str) -> str: ...
# def classify_intent(text: str) -> str: ...
