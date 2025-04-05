import os
import re
import logging
from typing import List, Dict, Optional, Any
import PyPDF2

logger = logging.getLogger(__name__)

# --- Константы ---
# Эти значения можно будет переопределить при вызове функций
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 100


def extract_text_from_pdf(filepath: str) -> Optional[str]:
    """
    Извлекает текст со всех страниц PDF-файла.

    Args:
        filepath: Путь к PDF-файлу.

    Returns:
        Извлеченный и объединенный текст или None в случае ошибки.
    """
    text = ""
    filename = os.path.basename(filepath)
    try:
        with open(filepath, 'rb') as pdf_file_obj:
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            num_pages = len(pdf_reader.pages)
            logger.debug(f"Чтение PDF '{filename}' - {num_pages} страниц.")
            for page_num in range(num_pages):
                try:
                    page_obj = pdf_reader.pages[page_num]
                    page_text = page_obj.extract_text()
                    if page_text:
                        page_text = re.sub(r'\s+', ' ', page_text).strip()
                        page_text = re.sub(r'\.', '', page_text).strip()
                        text += page_text
                    else:
                        logger.debug(
                            f"На странице {page_num + 1} в '{filename}' текст не найден.")
                except Exception as page_e:
                    logger.warning(
                        f"Ошибка при обработке страницы {page_num + 1} в файле '{filename}': {page_e}")
                    continue
        return text.strip() if text else None
    except FileNotFoundError:
        logger.error(f"Файл PDF не найден: {filepath}")
        return None
    except Exception as e:
        logger.error(
            f"Не удалось прочитать PDF {filepath}: {e}", exc_info=True)
        return None


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Разбивает текст на чанки по символам с заданным размером и перекрытием.

    Args:
        text: Исходный текст.
        chunk_size: Желаемый размер чанка в символах.
        chunk_overlap: Количество символов перекрытия между чанками.

    Returns:
        Список текстовых чанков.
    """
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        logger.warning(
            f"Перекрытие ({chunk_overlap}) больше или равно размеру чанка ({chunk_size}). Устанавливаю перекрытие в {chunk_size // 5}.")
        chunk_overlap = chunk_size // 5

    chunks = []
    start_index = 0
    text_len = len(text)

    while start_index < text_len:
        end_index = start_index + chunk_size
        chunk = text[start_index:end_index]
        chunks.append(chunk)

        next_start = start_index + chunk_size - chunk_overlap

        if next_start <= start_index:
            next_start = start_index + 1

        start_index = next_start

    return chunks


def load_and_chunk_pdfs(folder_path: str,
                        chunk_size: int = DEFAULT_CHUNK_SIZE,
                        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Загружает все PDF-файлы из указанной папки, извлекает текст
    и разбивает его на чанки.

    Args:
        folder_path: Путь к папке с PDF-файлами.
        chunk_size: Размер чанка для разбиения текста.
        chunk_overlap: Перекрытие чанков.

    Returns:
        Список словарей, где каждый словарь представляет чанк
        с полями 'id', 'text', 'source'.
    """
    all_chunks_data = []
    chunk_id_counter = 0
    logger.info(f"Начало обработки PDF из папки: {folder_path}")

    try:
        if not os.path.isdir(folder_path):
            logger.error(f"Указанный путь не является папкой: {folder_path}")
            return []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                filepath = os.path.join(folder_path, filename)
                logger.info(f"Обработка файла: {filename}")

                full_text = extract_text_from_pdf(filepath)
                if not full_text:
                    logger.warning(
                        f"Не удалось извлечь текст или файл пуст: {filename}")
                    continue

                text_chunks = chunk_text(full_text, chunk_size, chunk_overlap)
                if not text_chunks:
                    logger.warning(
                        f"Не удалось разбить текст на чанки: {filename}")
                    continue
                logger.info(
                    f"'{filename}' разбит на {len(text_chunks)} чанков (размер ~{chunk_size}, перекрытие ~{chunk_overlap}).")

                for i, chunk in enumerate(text_chunks):
                    chunk_data = {
                        "id": f"pdf_{filename}_chunk_{chunk_id_counter}",
                        "text": chunk,
                        "source": f"Документ: {filename}",
                        # Сюда можно добавить извлечение номера страницы, если маркер [Стр. X] найден в chunk
                    }
                    all_chunks_data.append(chunk_data)
                    chunk_id_counter += 1
            else:
                logger.debug(f"Пропущен не-PDF файл: {filename}")

    except FileNotFoundError:
        logger.error(f"Папка с PDF не найдена: {folder_path}")
        return []
    except Exception as e:
        logger.error(
            f"Непредвиденная ошибка при обработке PDF файлов в '{folder_path}': {e}", exc_info=True)
        return []

    logger.info(
        f"Обработка PDF завершена. Всего создано {len(all_chunks_data)} чанков.")
    return all_chunks_data


print(load_and_chunk_pdfs('knowledge_pdfs')[100])
