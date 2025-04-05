import database
import logging
import os
import html
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
)
from dotenv import load_dotenv

# Импортируем функции из нашего AI модуля
import ai_pipeline

load_dotenv()
# --- Конфигурация Бота ---
TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN")
# ID канала для логирования событий
LOG_CHANNEL_ID = os.environ.get("LOG_CHANNEL_ID")

# --- Настройка Логирования (делаем это здесь, чтобы было доступно везде) ---
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=log_level
)
# Устанавливаем уровень логгера для библиотеки telegram
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# --- Функция для отправки логов в канал ---

async def send_to_log_channel(context: ContextTypes.DEFAULT_TYPE, message: str, parse_mode=None) -> None:
    """Отправляет сообщение в канал логирования."""
    if not LOG_CHANNEL_ID:
        logger.warning("LOG_CHANNEL_ID не установлен. Логирование в канал отключено.")
        return
    
    try:
        await context.bot.send_message(
            chat_id=LOG_CHANNEL_ID,
            text=message,
            parse_mode=parse_mode
        )
    except Exception as e:
        logger.error(f"Ошибка отправки сообщения в канал логирования: {e}")


# --- Обработчики Команд ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение при команде /start."""
    user = update.effective_user
    logger.info(
        f"Пользователь {user.id} ({user.username}) запустил команду /start")
    await update.message.reply_html(
        f"Привет, {user.mention_html()}!\n"
        f"Я AI-ассистент по Порталу поставщиков. Задайте ваш вопрос.",
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
    
    # Логируем новый вопрос в канал
    log_question_message = (
        f"📥 <b>Новый вопрос</b>\n"
        f"Пользователь: {html.escape(user.full_name)} (ID: {user.id})\n"
        f"Имя пользователя: @{html.escape(user.username or 'отсутствует')}\n"
        f"Вопрос: <i>{html.escape(user_query)}</i>"
    )
    await send_to_log_channel(context, log_question_message, parse_mode="HTML")

    request_interaction_id = database.log_interaction(
        user_telegram_id=user.id,
        is_from_user=True,
        message_text=user_query
    )

    if not ai_pipeline.get_ai_status():
        logger.error("AI Core не инициализирован. Ответ невозможен.")
        error_text = "Извините, сервис временно недоступен. Попробуйте позже."
        database.log_interaction(
            user_telegram_id=user.id,
            is_from_user=False,
            message_text=error_text,
            request_interaction_id=request_interaction_id  # Связываем с запросом
        )
        
        # Логируем ошибку в канал
        log_error_message = (
            f"❌ <b>Ошибка</b>\n"
            f"Пользователь: {html.escape(user.full_name)} (ID: {user.id})\n"
            f"Причина: AI Core не инициализирован\n"
            f"Ответ: <i>{html.escape(error_text)}</i>"
        )
        await send_to_log_channel(context, log_error_message, parse_mode="HTML")
        
        await update.message.reply_text(error_text)
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
        # Используем HTML для лучшего форматирования ссылок и выделения
        response_parts.append(
            f"<blockquote>{best_result['text']}</blockquote>")  # Цитируем текст

        response_text = "\n".join(response_parts)

        # --- Добавляем кнопки Оценки ---
        keyboard = [
            [
                InlineKeyboardButton(
                    "👍 Нравится", callback_data=f"rate_up_{best_result.get('id', 'no_id')}_{request_interaction_id}"),
                InlineKeyboardButton(
                    "👎 Не нравится", callback_data=f"rate_down_{best_result.get('id', 'no_id')}_{request_interaction_id}"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        response_interaction_id = database.log_interaction(
            user_telegram_id=user.id,
            is_from_user=False,
            message_text=response_text,  # Записываем уже сформированный текст
            request_interaction_id=request_interaction_id,
            matched_kb_id=best_result.get('id'),
            similarity_score=best_result.get('similarity')
        )
        
        # Логируем ответ в канал
        log_answer_message = (
            f"📤 <b>Ответ бота</b>\n"
            f"Пользователь: {html.escape(user.full_name)} (ID: {user.id})\n"
            f"Вопрос: <i>{html.escape(user_query)}</i>\n"
            f"Найденный фрагмент ID: {best_result.get('id', 'unknown')}\n"
            f"Релевантность: {best_result.get('similarity', 0):.2f}\n"
            f"Ответ: <i>{html.escape(response_text)}</i>"
        )
        await send_to_log_channel(context, log_answer_message, parse_mode="HTML")

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

        response_interaction_id = database.log_interaction(
            user_telegram_id=user.id,
            is_from_user=False,
            message_text=response_text,
            request_interaction_id=request_interaction_id
        )
        
        # Логируем отсутствие ответа в канал
        log_no_answer_message = (
            f"🔍 <b>Ответ не найден</b>\n"
            f"Пользователь: {html.escape(user.full_name)} (ID: {user.id})\n"
            f"Вопрос: <i>{html.escape(user_query)}</i>\n"
            f"Ответ: <i>{html.escape(response_text)}</i>"
        )
        await send_to_log_channel(context, log_no_answer_message, parse_mode="HTML")

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

    # Парсим callback_data (пример: "rate_up_faq_1_123")
    parts = callback_data.split('_')
    action = parts[0]
    rate_type = parts[1]
    item_id = parts[2] if len(parts) > 2 else "no_id"  # ID фрагмента базы знаний
    interaction_to_rate_id = int(parts[3]) if len(parts) > 3 else 0  # ID взаимодействия

    if action == "rate" and interaction_to_rate_id:
        rating = 1 if rate_type == "up" else -1 if rate_type == "down" else 0
        if rating != 0:
            # Записываем оценку в БД
            success = database.log_rating(
                interaction_id=interaction_to_rate_id,
                user_telegram_id=user.id,
                rating_value=rating
            )
            
            # Логируем оценку в канал
            rating_text = "👍 Положительная" if rating == 1 else "👎 Отрицательная"
            log_rating_message = (
                f"⭐ <b>Оценка получена</b>\n"
                f"Пользователь: {html.escape(user.full_name)} (ID: {user.id})\n"
                f"Оценка: {rating_text}\n"
                f"ID взаимодействия: {interaction_to_rate_id}\n"
                f"ID фрагмента: {item_id}"
            )
            await send_to_log_channel(context, log_rating_message, parse_mode="HTML")
            
            if success:
                # Редактируем сообщение (убираем кнопки или добавляем текст)
                await query.edit_message_text(
                    text=query.message.text_html + "\n\n<i>Спасибо за вашу оценку!</i>",
                    parse_mode='HTML',
                    reply_markup=None  # Убираем клавиатуру
                )
            else:
                await query.answer("Не удалось сохранить оценку.", show_alert=True)
                
                # Логируем ошибку оценки
                log_rating_error = (
                    f"❌ <b>Ошибка сохранения оценки</b>\n"
                    f"Пользователь: {html.escape(user.full_name)} (ID: {user.id})\n"
                    f"ID взаимодействия: {interaction_to_rate_id}"
                )
                await send_to_log_channel(context, log_rating_error, parse_mode="HTML")
        else:
            logger.warning(f"Неверный тип оценки: {rate_type}")

    # except (IndexError, ValueError) as e:
    #     logger.error(f"Ошибка парсинга callback_data '{callback_data}': {e}")
    # except Exception as e:
    #     logger.error(
    #         f"Непредвиденная ошибка в button_callback: {e}", exc_info=True)


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет пользователю историю его последних взаимодействий."""
    user = update.effective_user
    logger.info(f"Пользователь {user.id} запросил историю.")

    history_records = database.get_user_history(
        user.id, limit=10)  # Запросим последние 10 "пар"

    if not history_records:
        await update.message.reply_text("Ваша история сообщений пока пуста.")
        return

    response_text = "<b>Ваша недавняя история:</b>\n\n"
    # Форматируем историю для вывода
    # Простой вариант: просто списком
    for record in history_records:
        prefix = "👤 Вы:" if record['is_from_user'] else "🤖 Бот:"
        timestamp_str = record['timestamp'].split(
            '.')[0]  # Убираем миллисекунды для краткости
        # Экранируем HTML-символы в тексте сообщения перед добавлением префикса
        safe_message = html.escape(record['message_text'])
        # Используем HTML форматирование
        response_text += f"<code>{timestamp_str}</code>\n{prefix} {safe_message}\n\n"
        # Лимит Telegram на длину сообщения - около 4096 символов.
        if len(response_text) > 3800:  # Оставляем запас
            await update.message.reply_html(response_text)
            response_text = ""  # Начинаем новое сообщение

    if response_text:  # Отправляем остаток, если есть
        await update.message.reply_html(response_text)


# --- Основная функция запуска ---

def main() -> None:
    """Главная функция для инициализации и запуска бота."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical(
            "TELEGRAM_BOT_TOKEN не установлен! Бот не может быть запущен.")
        return
        
    if not LOG_CHANNEL_ID:
        logger.warning("LOG_CHANNEL_ID не установлен! Логирование в канал отключено.")

    logger.info("Инициализация базы данных...")
    database.init_db()  # Вызываем инициализацию

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
    application.add_handler(CommandHandler("history", history_command))
    # Добавьте другие обработчики...

    # 4. Запуск бота
    logger.info("Запуск Telegram бота (polling)...")
    application.run_polling()
    logger.info("Telegram бот остановлен.")


if __name__ == "__main__":
    main()
