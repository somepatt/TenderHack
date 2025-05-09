import logging
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, constants
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
)
import html
from dotenv import load_dotenv

import database
import ai_pipeline

load_dotenv()

# --- Конфигурация Бота ---
TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN")
USE_LLM_GENERATION = os.environ.get(
    'USE_LLM_GENERATION', 'True').lower() == 'true'
USE_LLM_CLASSIFICATION = os.environ.get(
    'USE_LLM_CLASSIFICATION', 'True').lower() == 'true'
# LOG_CHANNEL_ID = os.environ.get("LOG_CHANNEL_ID")

# --- Настройка Логирования (делаем это здесь, чтобы было доступно везде) ---
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=log_level
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# # --- Функция для отправки логов в канал ---
# async def send_to_log_channel(context: ContextTypes.DEFAULT_TYPE, message: str, parse_mode=None) -> None:
#     """Отправляет сообщение в канал логирования."""
#     if not LOG_CHANNEL_ID:
#         logger.warning(
#             "LOG_CHANNEL_ID не установлен. Логирование в канал отключено.")
#         return

#     try:
#         await context.bot.send_message(
#             chat_id=LOG_CHANNEL_ID,
#             text=message,
#             parse_mode=parse_mode
#         )
#     except Exception as e:
#         logger.error(f"Ошибка отправки сообщения в канал логирования: {e}")


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
    user_query = update.message.text
    user = update.effective_user
    chat_id = update.message.chat_id
    logger.info(
        f"Получен запрос от {user.id} ({user.username}): '{user_query}'")

    retrieval_ok, generation_ok = ai_pipeline.get_ai_status()

    # --- 1. Классификация ---
    query_category = "Другое"
    if USE_LLM_CLASSIFICATION and generation_ok:
        await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
        query_category = ai_pipeline.classify_query_type_with_llm(user_query)
    elif not generation_ok:
        query_category = "Общие вопросы"

    # --- 2. Логирование запроса ---
    request_interaction_id = database.log_interaction(
        user.id, True, user_query, query_category)

    # --- 3. Выбор стратегии и ответ ---
    final_response_text = ""
    kb_id_for_log = None
    similarity_for_log = None
    reply_markup = None
    best_match_item = None  # Объявляем здесь

    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

    # --- Стратегия А: Поиск в Базе Знаний (CSV/PDF) ---
    if not retrieval_ok:
        final_response_text = "Извините, сервис временно недоступен (ошибка поиска)."
    else:
        best_match_item = ai_pipeline.retrieve_context(user_query)

        if best_match_item:
            top = best_match_item[0]
            kb_id_for_log = top.get('id')
            similarity_for_log = top.get('similarity')
            data_type = top.get('data_type')
            source = top.get('source', 'База Знаний')
            original_content = top.get('content', '')

            if data_type == 'csv':
                logger.info(
                    f"Найден готовый ответ в CSV (ID: {kb_id_for_log}).")

                response_parts = [
                    f"{html.escape(original_content)}",
                    f"<b>Источник:</b> {source}"
                ]
                final_response_text = "\n".join(response_parts)

            elif data_type == 'pdf':
                logger.info(
                    f"Найден релевантный чанк PDF (ID: {kb_id_for_log}).")
                generated_answer = None
                if USE_LLM_GENERATION and generation_ok:
                    logger.info(
                        "Пытаемся сгенерировать RAG ответ по PDF...")
                    generated_answer = ai_pipeline.generate_answer_with_llm(
                        user_query, best_match_item)

                if generated_answer:
                    final_response_text = generated_answer
                else:
                    logger.warning(
                        "Не удалось сгенерировать ответ LLM по PDF, показываем текст чанка.")
                    response_parts = [f"{html.escape(original_content)}",
                                      f"<b>Источник:</b> {source}"]
                    final_response_text = "\n".join(response_parts)
            else:
                logger.error(f"Неизвестный data_type: {data_type}")
                final_response_text = "Произошла внутренняя ошибка."

        else:
            logger.info("Релевантных данных в CSV/PDF не найдено.")
            if USE_LLM_GENERATION and generation_ok:
                final_response_text = ai_pipeline.generate_live_response_with_llm(
                    user_query, "Другое")
                if not final_response_text:
                    final_response_text = "К сожалению, не могу найти информацию и возникла ошибка."
            else:
                # --- Стратегия Б: "Живое" общение ---
                logger.info(
                    f"Категория '{query_category}' требует 'живого' ответа.")
                if USE_LLM_GENERATION and generation_ok:
                    final_response_text = ai_pipeline.generate_live_response_with_llm(
                        user_query, query_category)
                    if not final_response_text:
                        final_response_text = "Спасибо за сообщение. Ошибка обработки."
                else:
                    # Fallback без LLM
                    if query_category == "Жалобы":
                        final_response_text = "..."
                    elif query_category == "Обратная связь":
                        final_response_text = "..."
                    else:
                        final_response_text = "Спасибо за ваше сообщение!"

    # --- 5. Формирование кнопок и отправка ---
    if request_interaction_id and kb_id_for_log:
        keyboard = [[InlineKeyboardButton("👍", callback_data=f"rate_up_{request_interaction_id}"),
                     InlineKeyboardButton("👎", callback_data=f"rate_down_{request_interaction_id}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
    elif query_category == "Жалобы" or not best_match_item and query_category:
        keyboard = [[InlineKeyboardButton(
            "❓ Задать вопрос оператору", url="https://t.me/support_operator")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

    try:
        await update.message.reply_html(html.escape(final_response_text), reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Ошибка отправки: {e}")
# --- Обработчик Нажатий на Кнопки (Callback) ---


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает нажатия на инлайн-кнопки (например, оценки)."""
    query = update.callback_query
    await query.answer()

    try:
        callback_data = query.data
        user = query.from_user
        logger.info(
            f"Получен callback от {user.id} ({user.username}): {callback_data}")

        parts = callback_data.split('_')
        action = parts[0]
        rate_type = parts[1]
        item_id = parts[2] if len(parts) > 2 else "no_id"
        interaction_to_rate_id = int(parts[3]) if len(
            parts) > 3 else 0

        if action == "rate" and interaction_to_rate_id:
            rating = 1 if rate_type == "up" else -1 if rate_type == "down" else 0
            if rating != 0:
                # Записываем оценку в БД
                success = database.log_rating(
                    interaction_id=interaction_to_rate_id,
                    user_telegram_id=user.id,
                    rating_value=rating
                )

                if success:
                    await query.edit_message_text(
                        text=query.message.text_html + "\n\n<i>Спасибо за вашу оценку!</i>",
                        parse_mode='HTML',
                        reply_markup=None
                    )

                else:
                    await query.answer("Не удалось сохранить оценку.", show_alert=True)

                    log_rating_error = (
                        f"❌ <b>Ошибка сохранения оценки</b>\n"
                        f"Пользователь: {html.escape(user.full_name)} (ID: {user.id})\n"
                        f"ID взаимодействия: {interaction_to_rate_id}"
                    )
                    await send_to_log_channel(context, log_rating_error, parse_mode="HTML")
            else:
                logger.warning(f"Неверный тип оценки: {rate_type}")

    except (IndexError, ValueError) as e:
        logger.error(f"Ошибка парсинга callback_data '{callback_data}': {e}")
    except Exception as e:
        logger.error(
            f"Непредвиденная ошибка в button_callback: {e}", exc_info=True)


# async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Отправляет пользователю историю его последних взаимодействий."""
#     user = update.effective_user
#     logger.info(f"Пользователь {user.id} запросил историю.")

#     # history_records = database.get_user_history(
#     #     user.id, limit=10)  # Запросим последние 10 "пар"

#     if not history_records:
#         await update.message.reply_text("Ваша история сообщений пока пуста.")
#         return

#     response_text = "<b>Ваша недавняя история:</b>\n\n"
#     # Форматируем историю для вывода
#     # Простой вариант: просто списком
#     for record in history_records:
#         prefix = "👤 Вы:" if record['is_from_user'] else "🤖 Бот:"
#         timestamp_str = record['timestamp'].split(
#             '.')[0]  # Убираем миллисекунды для краткости
#         # Экранируем HTML-символы в тексте сообщения перед добавлением префикса
#         safe_message = html.escape(record['message_text'])
#         # Используем HTML форматирование
#         response_text += f"<code>{timestamp_str}</code>\n{prefix} {safe_message}\n\n"
#         # Лимит Telegram на длину сообщения - около 4096 символов.
#         if len(response_text) > 3800:  # Оставляем запас
#             await update.message.reply_html(response_text)
#             response_text = ""  # Начинаем новое сообщение

#     if response_text:  # Отправляем остаток, если есть
#         await update.message.reply_html(response_text)


# --- Основная функция запуска ---

def main() -> None:
    """Главная функция для инициализации и запуска бота."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical(
            "TELEGRAM_BOT_TOKEN не установлен! Бот не может быть запущен.")
        return

    # if not LOG_CHANNEL_ID:
    #     logger.warning(
    #         "LOG_CHANNEL_ID не установлен! Логирование в канал отключено.")

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
    # application.add_handler(CommandHandler("history", history_command))
    # Добавьте другие обработчики...

    # 4. Запуск бота
    logger.info("Запуск Telegram бота (polling)...")
    application.run_polling()
    logger.info("Telegram бот остановлен.")


if __name__ == "__main__":
    main()
