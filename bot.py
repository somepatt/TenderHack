import logging
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
)

# Импортируем функции из нашего AI модуля
import ai_pipeline

# --- Конфигурация Бота ---
TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN")

# --- Настройка Логирования (делаем это здесь, чтобы было доступно везде) ---
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=log_level
)
# Устанавливаем уровень логгера для библиотеки telegram
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# --- Обработчики Команд ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение при команде /start."""
    user = update.effective_user
    logger.info(
        f"Пользователь {user.id} ({user.username}) запустил команду /start")
    await update.message.reply_html(
        f"Привет, {user.mention_html()}!\n"
        f"Я AI-ассистент по Порталу поставщиков. Задайте ваш вопрос.",
        # disable_web_page_preview=True # Если не хотим превью ссылок
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет сообщение с помощью по команде /help."""
    logger.info(f"Пользователь {update.effective_user.id} запросил помощь")
    help_text = (
        "Просто напишите ваш вопрос, и я постараюсь найти ответ в базе знаний.\n\n"
        "Примеры запросов:\n"
        "- Как подать заявку на котировочную сессию?\n"
        "- Нужна ли электронная подпись для регистрации?\n"
        "- Ошибка при подписании: не удалось построить цепочку сертификатов"
        # Добавьте другие команды, если они появятся (/history и т.д.)
    )
    await update.message.reply_text(help_text)


# --- Обработчик Сообщений ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовое сообщение пользователя, ищет ответ и отправляет его."""
    user_query = update.message.text
    user = update.effective_user
    chat_id = update.message.chat_id

    logger.info(
        f"Получен запрос от {user.id} ({user.username}): '{user_query}'")

    if not ai_pipeline.get_ai_status():
        logger.error("AI Core не инициализирован. Ответ невозможен.")
        await update.message.reply_text("Извините, сервис временно недоступен. Попробуйте позже.")
        return

    # --- Сюда можно добавить предобработку: исправление опечаток ---
    # corrected_query = ai_pipeline.correct_spelling(user_query)
    # search_results = ai_pipeline.find_relevant_knowledge(corrected_query)
    search_results = ai_pipeline.find_relevant_knowledge(user_query)

    # --- Формируем ответ ---
    if search_results:
        # Берем лучший результат
        best_result = search_results[0]
        response_parts = []
        response_parts.append(
            f"Нашел ответ (схожесть: {best_result['similarity']:.2f}):\n"
            # f"_(Возможно, по запросу: '{corrected_query}')_\n" # Если используем исправление
        )
        # Используем HTML для лучшего форматирования ссылок и выделения
        response_parts.append(
            f"<blockquote>{best_result['text']}</blockquote>")  # Цитируем текст

        response_text = "\n".join(response_parts)

        # --- Добавляем кнопки Оценки ---
        keyboard = [
            [
                InlineKeyboardButton(
                    "👍 Нравится", callback_data=f"rate_up_{best_result.get('id', 'no_id')}"),
                InlineKeyboardButton(
                    "👎 Не нравится", callback_data=f"rate_down_{best_result.get('id', 'no_id')}"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_html(response_text, reply_markup=reply_markup)
        log_data = {"query": user_query, "result_id": best_result.get(
            'id'), "similarity": best_result['similarity']}

    else:
        response_text = "К сожалению, я не смог найти точный ответ в базе знаний. Попробуйте переформулировать ваш вопрос."
        # --- Кнопка связи с оператором ---
        keyboard = [
            # Замените на реальный контакт
            [InlineKeyboardButton(
                "❓ Задать вопрос оператору", url="https://t.me/YOUR_SUPPORT_CONTACT")]
        ]
        # Или можно сделать callback_data="ask_operator" и обработать его
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(response_text, reply_markup=reply_markup)
        log_data = {"query": user_query, "result_id": None, "similarity": 0.0}

    # --- Логирование Взаимодействия (замените на вашу реальную БД/файл) ---
    logger.info(f"Interaction log for user {user.id}: {log_data}")
    # save_interaction_to_db(user.id, log_data) # Ваша функция сохранения


# --- Обработчик Нажатий на Кнопки (Callback) ---

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает нажатия на инлайн-кнопки (например, оценки)."""
    query = update.callback_query
    await query.answer()  # Обязательно нужно ответить на колбэк

    callback_data = query.data
    user = query.from_user
    logger.info(
        f"Получен callback от {user.id} ({user.username}): {callback_data}")

    # Парсим callback_data (пример: "rate_up_faq_1")
    parts = callback_data.split('_')
    action = parts[0]
    rate_type = parts[1] if len(parts) > 1 else None
    item_id = "_".join(parts[2:]) if len(
        parts) > 2 else None  # ID может содержать '_'

    if action == "rate" and rate_type and item_id:
        rating = 1 if rate_type == "up" else -1 if rate_type == "down" else 0
        logger.info(
            f"Пользователь {user.id} оценил ответ {item_id} как {rating}")
        # --- Сохранение оценки в БД ---
        # save_rating_to_db(user.id, item_id, rating) # Ваша функция

        # Можно отредактировать сообщение, убрав кнопки или добавив текст "Спасибо за оценку!"
        await query.edit_message_text(text=query.message.text_html + "\n\n<i>Спасибо за вашу оценку!</i>", parse_mode='HTML')
        # Или просто убрать кнопки
        # await query.edit_message_reply_markup(reply_markup=None)

    # Добавьте обработку других кнопок, если нужно (ask_operator и т.д.)


# --- Основная функция запуска ---

def main() -> None:
    """Главная функция для инициализации и запуска бота."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical(
            "TELEGRAM_BOT_TOKEN не установлен! Бот не может быть запущен.")
        return

    # 1. Инициализация AI ядра
    logger.info("Инициализация AI ядра...")
    if not ai_pipeline.initialize_ai_core():
        logger.critical("Ошибка инициализации AI ядра! Запуск бота отменен.")
        return
    logger.info("AI ядро успешно инициализировано.")

    # 2. Создание и настройка приложения бота
    logger.info("Настройка Telegram приложения...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # 3. Регистрация обработчиков
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    # Добавьте другие обработчики...

    # 4. Запуск бота
    logger.info("Запуск Telegram бота (polling)...")
    application.run_polling()
    logger.info("Telegram бот остановлен.")


if __name__ == "__main__":
    main()
