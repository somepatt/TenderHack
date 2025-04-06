import sqlite3
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# --- Конфигурация ---
DATABASE_FILE = os.environ.get('DATABASE_FILE')

logger = logging.getLogger(__name__)

# --- Функции для работы с БД ---


def _get_db_connection() -> sqlite3.Connection:
    """Устанавливает соединение с БД SQLite."""
    try:
        conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        logger.debug(f"Установлено соединение с БД: {DATABASE_FILE}")
        return conn
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка подключения к базе данных {DATABASE_FILE}: {e}", exc_info=True)
        raise


def init_db():
    """Инициализирует базу данных: создает таблицы, если их нет."""
    logger.info(f"Инициализация базы данных: {DATABASE_FILE}...")
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()

            # Таблица interactions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_telegram_id INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_from_user INTEGER NOT NULL, -- 1 for True, 0 for False
                    message_text TEXT NOT NULL,
                    query_category TEXT,
                    request_interaction_id INTEGER, -- FK to interactions(interaction_id)
                    matched_kb_id TEXT,
                    similarity_score REAL,
                    assigned_theme_id INTEGER,      -- FK to request_themes(theme_id)
                    FOREIGN KEY (request_interaction_id) REFERENCES interactions(interaction_id) ON DELETE SET NULL,
                    FOREIGN KEY (assigned_theme_id) REFERENCES request_themes(theme_id) ON DELETE SET NULL
                );
            """)
            logger.debug("Таблица 'interactions' проверена/создана.")

            # Таблица ratings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id INTEGER NOT NULL, -- FK to interactions(interaction_id) - ID ответа бота
                    user_telegram_id INTEGER NOT NULL,
                    rating_value INTEGER NOT NULL, -- e.g., 1 for like, -1 for dislike
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (interaction_id) REFERENCES interactions(interaction_id) ON DELETE CASCADE
                );
            """)
            logger.debug("Таблица 'ratings' проверена/создана.")

            # Таблица request_themes (для аналитики)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS request_themes (
                    theme_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    theme_name TEXT UNIQUE NOT NULL,
                    query_count INTEGER DEFAULT 0,
                    average_rating REAL,
                    last_updated DATETIME
                );
            """)
            logger.debug("Таблица 'request_themes' проверена/создана.")

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_user_time ON interactions (user_telegram_id, timestamp);")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_request ON interactions (request_interaction_id);")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_theme ON interactions (assigned_theme_id);")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_ratings_interaction ON ratings (interaction_id);")
            logger.debug("Индексы проверены/созданы.")

            conn.commit()
            logger.info("Инициализация базы данных завершена успешно.")

    except sqlite3.Error as e:
        logger.error(
            f"Ошибка при инициализации базы данных: {e}", exc_info=True)


def log_interaction(
    user_telegram_id: int,
    is_from_user: bool,
    message_text: str,
    query_category: Optional[str] = None,
    request_interaction_id: Optional[int] = None,
    matched_kb_id: Optional[str] = None,
    similarity_score: Optional[float] = None,
    assigned_theme_id: Optional[int] = None
) -> Optional[int]:
    """
    Записывает взаимодействие (сообщение пользователя или бота) в БД.

    Возвращает ID созданной записи взаимодействия или None в случае ошибки.
    """

    effective_category = query_category if is_from_user else None
    sql = """
        INSERT INTO interactions
        (user_telegram_id, is_from_user, message_text, query_category, request_interaction_id, matched_kb_id, similarity_score, assigned_theme_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        user_telegram_id,
        1 if is_from_user else 0,
        message_text,
        effective_category,
        request_interaction_id,
        matched_kb_id,
        similarity_score,
        assigned_theme_id
    )
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            last_id = cursor.lastrowid
            conn.commit()
            logger.debug(
                f"Записано взаимодействие ID: {last_id} для пользователя {user_telegram_id}")
            return last_id
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка записи взаимодействия для пользователя {user_telegram_id}: {e}", exc_info=True)
        return None


def log_rating(interaction_id: int, user_telegram_id: int, rating_value: int) -> bool:
    """
    Записывает оценку пользователя для ответа бота.

    Возвращает True при успехе, False при ошибке.
    """
    # Опционально: можно сначала проверить, не оценивал ли этот пользователь уже это сообщение
    sql = """
        INSERT INTO ratings (interaction_id, user_telegram_id, rating_value)
        VALUES (?, ?, ?)
    """
    params = (interaction_id, user_telegram_id, rating_value)
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            logger.info(
                f"Записана оценка {rating_value} для ответа {interaction_id} от пользователя {user_telegram_id}")
            return True
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка записи оценки для ответа {interaction_id} от пользователя {user_telegram_id}: {e}", exc_info=True)
        return False


def get_user_history(user_telegram_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Получает историю последних N взаимодействий для указанного пользователя.

    Возвращает список словарей, представляющих сообщения.
    """
    effective_limit = limit * 2 + 10
    sql = """
        SELECT
            interaction_id,
            timestamp,
            is_from_user, -- 1 или 0
            message_text,
            query_category,
            request_interaction_id,
            matched_kb_id,
            similarity_score
            -- Можно добавить получение оценки, если нужно показывать ее в истории
            -- (SELECT rating_value FROM ratings r WHERE r.interaction_id = i.interaction_id) as rating
        FROM interactions AS i -- Даем таблице псевдоним 'i'
        WHERE user_telegram_id = ?
        ORDER BY timestamp DESC -- Сначала самые новые
        LIMIT ?
    """
    params = (user_telegram_id, effective_limit)
    history = []
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            # Преобразуем строки в словари и boolean обратно
            for row in rows:
                interaction_dict = dict(row)
                interaction_dict['is_from_user'] = bool(
                    interaction_dict['is_from_user'])
                history.append(interaction_dict)
            # Возвращаем в хронологическом порядке (от старых к новым)
            history.reverse()
            logger.debug(
                f"Получено {len(history)} записей истории для пользователя {user_telegram_id}")
            return history
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка получения истории для пользователя {user_telegram_id}: {e}", exc_info=True)
        return []

# --- Функции для Аналитики (можно реализовать позже) ---


def assign_theme_to_interaction(interaction_id: int, theme_id: int):
    """Присваивает ID темы конкретному запросу пользователя."""
    sql = "UPDATE interactions SET assigned_theme_id = ? WHERE interaction_id = ?"
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (theme_id, interaction_id))
            conn.commit()
            logger.debug(
                f"Взаимодействию {interaction_id} присвоена тема {theme_id}")
            return True
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка присвоения темы {theme_id} взаимодействию {interaction_id}: {e}", exc_info=True)
        return False


def update_theme_stats(theme_id: int, query_count: int, average_rating: Optional[float]):
    """Обновляет статистику для темы в таблице request_themes."""
    sql = """
        UPDATE request_themes
        SET query_count = ?, average_rating = ?, last_updated = CURRENT_TIMESTAMP
        WHERE theme_id = ?
    """
    params = (query_count, average_rating, theme_id)
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            logger.debug(f"Обновлена статистика для темы {theme_id}")
            return True
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка обновления статистики для темы {theme_id}: {e}", exc_info=True)
        return False


def get_or_create_theme(theme_name: str) -> Optional[int]:
    """Получает ID темы по имени или создает новую, если не найдена."""
    # Сначала пытаемся найти
    sql_find = "SELECT theme_id FROM request_themes WHERE theme_name = ?"
    sql_create = "INSERT INTO request_themes (theme_name) VALUES (?)"
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql_find, (theme_name,))
            result = cursor.fetchone()
            if result:
                return result['theme_id']
            else:
                # Создаем новую
                cursor.execute(sql_create, (theme_name,))
                last_id = cursor.lastrowid
                conn.commit()
                logger.info(
                    f"Создана новая тема '{theme_name}' с ID: {last_id}")
                return last_id
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка при получении/создании темы '{theme_name}': {e}", exc_info=True)
        return None
