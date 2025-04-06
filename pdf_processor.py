import os
import re
import logging
from typing import List, Dict, Optional, Any, Tuple
import PyPDF2

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 100


def extract_pages_from_pdf(filepath: str) -> Optional[List[Tuple[int, str]]]:
    """
    Извлекает текст с каждой страницы PDF-файла.

    Args:
        filepath: Путь к PDF-файлу.

    Returns:
        Список кортежей (номер_страницы, текст_страницы) или None в случае ошибки.
        Номер страницы начинается с 1.
    """
    pages_data = []
    filename = os.path.basename(filepath)
    try:
        with open(filepath, 'rb') as pdf_file_obj:
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            num_pages = len(pdf_reader.pages)
            logger.debug(f"Чтение PDF '{filename}' - {num_pages} страниц.")
            for page_num in range(num_pages):
                current_page_number = page_num + 1
                try:
                    page_obj = pdf_reader.pages[page_num]
                    page_text = page_obj.extract_text()
                    if page_text:
                        page_text = re.sub(r'\s+', ' ', page_text).strip()
                        page_text = re.sub(r'\.', '', page_text).strip()
                        if page_text:
                            pages_data.append((current_page_number, page_text))
                except Exception as page_e:
                    logger.warning(
                        f"Ошибка при обработке страницы {current_page_number} в файле '{filename}': {page_e}")
                    continue
        return pages_data if pages_data else None
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
    Загружает все PDF-файлы из папки, извлекает текст постранично
    и разбивает текст каждой страницы на чанки, сохраняя номер страницы.
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

                pages_content = extract_pages_from_pdf(filepath)
                if not pages_content:
                    logger.warning(
                        f"Не удалось извлечь страницы или файл пуст: {filename}")
                    continue

                for page_number, page_text in pages_content:
                    if not page_text:
                        continue

                    text_chunks = chunk_text(
                        page_text, chunk_size, chunk_overlap)
                    if not text_chunks:
                        logger.warning(
                            f"Не удалось разбить текст страницы {page_number} на чанки: {filename}")
                        continue

                    logger.debug(
                        f"Страница {page_number} файла '{filename}' разбита на {len(text_chunks)} чанков.")

                    for i, chunk in enumerate(text_chunks):
                        chunk_data = {
                            "id": f"pdf_{filename}_p{page_number}_chunk_{chunk_id_counter}",
                            "text": chunk,
                            "source": f"Документ: {filename}",
                            "page": page_number
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
            f"Непредвиденная ошибка при обработке PDF: {e}", exc_info=True)
        return []

    logger.info(
        f"Обработка PDF завершена. Всего создано {len(all_chunks_data)} чанков.")
    return all_chunks_data
