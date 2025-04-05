import logging
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv
import sqlite3
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment variables
ADMIN_BOT_TOKEN = os.getenv("ADMIN_BOT_TOKEN")
DATABASE_FILE = os.getenv("DATABASE_FILE", "db.sqlite3")

def get_db_connection() -> sqlite3.Connection:
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "Привет! Я админ-бот для системы поддержки Портала поставщиков.\n"
        "Используйте /stats для получения статистики."
    )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send statistics when the command /stats is issued."""
    try:
        # Get statistics from the database
        total_requests = get_total_requests()
        likes, dislikes = get_ratings_count()
        top_questions = get_top_liked_questions()
        
        # Format the message
        message = f"📊 *Статистика системы*\n\n"
        message += f"*Общее количество запросов:* {total_requests}\n\n"
        message += f"*Оценки пользователей:*\n"
        message += f"👍 Лайки: {likes}\n"
        message += f"👎 Дизлайки: {dislikes}\n\n"
        
        message += "*Топ-5 популярных вопросов:*\n"
        if top_questions:
            for i, (text, likes) in enumerate(top_questions, 1):
                # Truncate long questions
                short_text = text[:100] + "..." if len(text) > 100 else text
                message += f"{i}. \"{short_text}\" - {likes} 👍\n"
        else:
            message += "Пока нет оцененных вопросов.\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error generating stats: {e}", exc_info=True)
        await update.message.reply_text(f"Ошибка при получении статистики: {e}")

def get_total_requests() -> int:
    """Get the total number of user requests from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM interactions 
            WHERE is_from_user = 1
        """)
        result = cursor.fetchone()
        return result['count'] if result else 0

def get_ratings_count() -> Tuple[int, int]:
    """Get the count of likes and dislikes from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Get likes count
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM ratings 
            WHERE rating_value > 0
        """)
        likes = cursor.fetchone()['count']
        
        # Get dislikes count
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM ratings 
            WHERE rating_value < 0
        """)
        dislikes = cursor.fetchone()['count']
        
        return likes, dislikes

def get_top_liked_questions() -> List[Tuple[str, int]]:
    """Get the top 5 questions by number of likes."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                req.message_text,
                COUNT(r.rating_id) as like_count
            FROM 
                interactions as resp
            JOIN 
                interactions as req ON resp.request_interaction_id = req.interaction_id
            JOIN 
                ratings as r ON r.interaction_id = resp.interaction_id
            WHERE 
                req.is_from_user = 1 AND
                r.rating_value > 0
            GROUP BY 
                req.message_text
            ORDER BY 
                like_count DESC
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        return [(row['message_text'], row['like_count']) for row in results]

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = ApplicationBuilder().token(ADMIN_BOT_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats))

    # Start the Bot
    logger.info("Starting Admin Bot...")
    application.run_polling()

if __name__ == '__main__':
    if not ADMIN_BOT_TOKEN:
        logger.error("ADMIN_BOT_TOKEN not set in environment variables!")
    else:
        main() 