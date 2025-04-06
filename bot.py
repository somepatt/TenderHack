import logging
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, constants
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
)
import html
from dotenv import load_dotenv


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

# async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     user_query = update.message.text
#     user = update.effective_user
#     chat_id = update.message.chat_id
#     logger.info(
#         f"Получен запрос от {user.id} ({user.username}): '{user_query}'")

#     # Логируем новый вопрос в канал
#     log_question_message = (
#         f"📥 <b>Новый вопрос</b>\n"
#         f"Пользователь: {html.escape(user.full_name)} (ID: {user.id})\n"
#         f"Имя пользователя: @{html.escape(user.username or 'отсутствует')}\n"
#         f"Вопрос: <i>{html.escape(user_query)}</i>"
#     )
#     await send_to_log_channel(context, log_question_message, parse_mode="HTML")

#     retrieval_ok, generation_ok = ai_pipeline.get_ai_status()
#     # request_interaction_id = database.log_interaction(
#     #     user_telegram_id=user.id,
#     #     is_from_user=True,
#     #     message_text=user_query
#     # )

#     if not ai_pipeline.get_ai_status():
#         logger.error("AI Core не инициализирован. Ответ невозможен.")
#         error_text = "Извините, сервис временно недоступен. Попробуйте позже."
#         # database.log_interaction(
#         #     user_telegram_id=user.id,
#         #     is_from_user=False,
#         #     message_text=error_text,
#         #     request_interaction_id=request_interaction_id  # Связываем с запросом
#         # )

#         # Логируем ошибку в канал
#         log_error_message = (
#             f"❌ <b>Ошибка</b>\n"
#             f"Пользователь: {html.escape(user.full_name)} (ID: {user.id})\n"
#             f"Причина: AI Core не инициализирован\n"
#             f"Ответ: <i>{html.escape(error_text)}</i>"
#         )
#         await send_to_log_channel(context, log_error_message, parse_mode="HTML")

#         await update.message.reply_text(error_text)
#         return

#     search_results = ai_pipeline.retrieve_context(user_query)

#     # --- Формируем ответ ---
#     if search_results:
#         # Берем лучший результат
#         best_result = search_results
#         response_parts = []
#         # Используем HTML для лучшего форматирования ссылок и выделения
#         response_parts.append(
#             f"<blockquote>{best_result['text']}</blockquote>")  # Цитируем текст

#         response_text = "\n".join(response_parts)

#         # --- Добавляем кнопки Оценки ---
#         keyboard = [
#             [
#                 InlineKeyboardButton(
#                     "👍 Нравится", callback_data=f"rate_up_{best_result.get('id', 'no_id')}_{request_interaction_id}"),
#                 InlineKeyboardButton(
#                     "👎 Не нравится", callback_data=f"rate_down_{best_result.get('id', 'no_id')}_{request_interaction_id}"),
#             ]
#         ]
#         reply_markup = InlineKeyboardMarkup(keyboard)

#         # response_interaction_id = database.log_interaction(
#         #     user_telegram_id=user.id,
#         #     is_from_user=False,
#         #     message_text=response_text,  # Записываем уже сформированный текст
#         #     request_interaction_id=request_interaction_id,
#         #     matched_kb_id=best_result.get('id'),
#         #     similarity_score=best_result.get('similarity')
#         # )

#         # Логируем ответ в канал
#         log_answer_message = (
#             f"📤 <b>Ответ бота</b>\n"
#             f"Пользователь: {html.escape(user.full_name)} (ID: {user.id})\n"
#             f"Вопрос: <i>{html.escape(user_query)}</i>\n"
#             f"Найденный фрагмент ID: {best_result.get('id', 'unknown')}\n"
#             f"Релевантность: {best_result.get('similarity', 0):.2f}\n"
#             f"Ответ: <i>{html.escape(response_text)}</i>"
#         )
#         await send_to_log_channel(context, log_answer_message, parse_mode="HTML")

#     # --- 1. Классификация ---
#     query_category = "Другое"
#     if USE_LLM_CLASSIFICATION and generation_ok:
#         await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
#         query_category = ai_pipeline.classify_query_type_with_llm(user_query)
#     elif not generation_ok:
#         query_category = "Общие вопросы"

#     # --- 2. Логирование запроса ---
#     # request_interaction_id = database.log_interaction(
#     #     user.id, True, user_query, query_category)

#     # --- 3. Выбор стратегии и ответ ---
#     final_response_text = ""
#     kb_id_for_log = None
#     similarity_for_log = None
#     reply_markup = None
#     best_match_item = None  # Объявляем здесь

#     await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

#     # --- Стратегия А: Поиск в Базе Знаний (CSV/PDF) ---
#     if query_category in ai_pipeline.SEARCH_KB_CATEGORIES:
#         logger.info(
#             f"Категория '{query_category}' требует поиска в БЗ (CSV/PDF).")
#         if not retrieval_ok:
#             final_response_text = "Извините, сервис временно недоступен (ошибка поиска)."
#         else:
#             best_match_item = ai_pipeline.retrieve_context(user_query)

#             if best_match_item:
#                 kb_id_for_log = best_match_item.get('id')
#                 similarity_for_log = best_match_item.get('similarity')
#                 data_type = best_match_item.get('data_type')
#                 source = best_match_item.get('source', 'База Знаний')
#                 # Оригинальный контент (ответ CSV или чанк PDF)
#                 original_content = best_match_item.get('content', '')

#                 # --- Если нашли ответ в CSV ---
#                 if data_type == 'csv':
#                     logger.info(
#                         f"Найден готовый ответ в CSV (ID: {kb_id_for_log}).")
#                     paraphrased_answer = None
#                     # --- Попытка перефразирования ---
#                     if PARAPHRASE_CSV_ANSWERS and generation_ok:
#                         logger.info("Пытаемся перефразировать ответ из CSV...")
#                         paraphrased_answer = ai_pipeline.paraphrase_text_with_llm(
#                             original_content)

#                     if paraphrased_answer:
#                         # Используем перефразированный ответ
#                         logger.info("Используем перефразированный ответ.")
#                         response_parts = [
#                             f"Нашел ответ в базе Q&A (схожесть вопроса: {similarity_for_log:.2f}):",
#                             f"<blockquote>{telegram.helpers.escape_html(paraphrased_answer)}</blockquote>",
#                             f"<b>Источник:</b> {source}"
#                         ]
#                         final_response_text = "\n".join(response_parts)
#                     else:
#                         # Используем оригинальный ответ (fallback)
#                         logger.info(
#                             "Используем оригинальный ответ из CSV (перефразирование не удалось или отключено).")
#                         response_parts = [
#                             f"Нашел ответ в базе Q&A (схожесть вопроса: {similarity_for_log:.2f}):",
#                             "<b>Ответ:</b>",
#                             f"<blockquote>{telegram.helpers.escape_html(original_content)}</blockquote>",
#                             f"<b>Источник:</b> {source}"
#                         ]
#                         final_response_text = "\n".join(response_parts)

#                 # --- Если нашли релевантный чанк в PDF ---
#                 elif data_type == 'pdf':
#                     logger.info(
#                         f"Найден релевантный чанк PDF (ID: {kb_id_for_log}).")
#                     generated_answer = None
#                     # Пытаемся сгенерировать ответ RAG
#                     if USE_LLM_GENERATION and generation_ok:
#                         logger.info(
#                             "Пытаемся сгенерировать RAG ответ по PDF...")
#                         generated_answer = ai_pipeline.generate_answer_with_llm(
#                             user_query, best_match_item)

#                     if generated_answer:
#                         final_response_text = generated_answer
#                     else:
#                         # Fallback: Показываем сам чанк PDF
#                         logger.warning(
#                             "Не удалось сгенерировать ответ LLM по PDF, показываем текст чанка.")
#                         response_parts = [f"Нашел релевантный фрагмент в документе '{source}' (схожесть: {similarity_for_log:.2f}):",
#                                           f"<blockquote>{telegram.helpers.escape_html(original_content)}</blockquote>",
#                                           f"<b>Источник:</b> {source}"]
#                         final_response_text = "\n".join(response_parts)
#                 else:
#                     logger.error(f"Неизвестный data_type: {data_type}")
#                     final_response_text = "Произошла внутренняя ошибка."

#             else:  # Ничего не найдено в БЗ
#                 logger.info("Релевантных данных в CSV/PDF не найдено.")
#                 # ... (код генерации "из головы" или "не найдено", без изменений) ...
#                 if USE_LLM_GENERATION and generation_ok:
#                     final_response_text = ai_pipeline.generate_live_response_with_llm(
#                         user_query, "Другое")
#                     if final_response_text:
#                         final_response_text += "\n\n_(Ответ сгенерирован без базы знаний)_"
#                     else:
#                         final_response_text = "К сожалению, не могу найти информацию и возникла ошибка."
#                 else:
#                     final_response_text = "К сожалению, я не смог найти ответ в базе знаний."

#     # --- Стратегия Б: "Живое" общение ---
#     else:
#         # ... (код без изменений) ...
#         logger.info(f"Категория '{query_category}' требует 'живого' ответа.")
#         if USE_LLM_GENERATION and generation_ok:
#             final_response_text = ai_pipeline.generate_live_response_with_llm(
#                 user_query, query_category)
#             if not final_response_text:
#                 final_response_text = "Спасибо за сообщение. Ошибка обработки."
#         else:
#             # Fallback без LLM
#             if query_category == "Жалобы":
#                 final_response_text = "..."
#             elif query_category == "Обратная связь":
#                 final_response_text = "..."
#             else:
#                 final_response_text = "Спасибо за ваше сообщение!"
#         response_text = "К сожалению, я не смог найти точный ответ в базе знаний. Попробуйте переформулировать ваш вопрос."
#         # --- Кнопка связи с оператором ---
#         keyboard = [
#             [InlineKeyboardButton(
#                 "❓ Задать вопрос оператору", url="")]
#         ]
#         # Или можно сделать callback_data="ask_operator" и обработать его
#         reply_markup = InlineKeyboardMarkup(keyboard)

#         # response_interaction_id = database.log_interaction(
#         #     user_telegram_id=user.id,
#         #     is_from_user=False,
#         #     message_text=response_text,
#         #     request_interaction_id=request_interaction_id
#         # )

#         # Логируем отсутствие ответа в канал
#         log_no_answer_message = (
#             f"🔍 <b>Ответ не найден</b>\n"
#             f"Пользователь: {html.escape(user.full_name)} (ID: {user.id})\n"
#             f"Вопрос: <i>{html.escape(user_query)}</i>\n"
#             f"Ответ: <i>{html.escape(response_text)}</i>"
#         )
#         await send_to_log_channel(context, log_no_answer_message, parse_mode="HTML")

#     # --- 4. Логирование ОТВЕТА бота ---
#     # ... (код без изменений, final_response_text теперь может быть перефразированным) ...
#     # response_interaction_id = database.log_interaction(
#     #     user.id, False, final_response_text, None, request_interaction_id,
#     #     kb_id_for_log, similarity_for_log
#     # )

#     # --- 5. Формирование кнопок и отправка ---
#     # ... (код без изменений) ...
#     if response_interaction_id and kb_id_for_log:
#         keyboard = [[InlineKeyboardButton("👍", callback_data=f"rate_up_{response_interaction_id}"),
#                      InlineKeyboardButton("👎", callback_data=f"rate_down_{response_interaction_id}")]]
#         reply_markup = InlineKeyboardMarkup(keyboard)
#     elif query_category == "Жалобы" or not best_match_item and query_category in ai_pipeline.SEARCH_KB_CATEGORIES:
#         keyboard = [[InlineKeyboardButton(
#             "❓ Задать вопрос оператору", url="...")]]
#         reply_markup = InlineKeyboardMarkup(keyboard)

#     try:
#         await update.message.reply_html(final_response_text, reply_markup=reply_markup)
#     except Exception as e:
#         logger.error(f"Ошибка отправки: {e}")  # Fallback отправка


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
    if response_interaction_id and kb_id_for_log:
        keyboard = [[InlineKeyboardButton("👍", callback_data=f"rate_up_{response_interaction_id}"),
                     InlineKeyboardButton("👎", callback_data=f"rate_down_{response_interaction_id}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
    elif query_category == "Жалобы" or not best_match_item and query_category in ai_pipeline.SEARCH_KB_CATEGORIES:
        keyboard = [[InlineKeyboardButton(
            "❓ Задать вопрос оператору", url="...")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

    try:
        await update.message.reply_html(final_response_text, reply_markup=reply_markup)
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

                # # Логируем оценку в канал
                # rating_text = "👍 Положительная" if rating == 1 else "👎 Отрицательная"
                # log_rating_message = (
                #     f"⭐ <b>Оценка получена</b>\n"
                #     f"Пользователь: {html.escape(user.full_name)} (ID: {user.id})\n"
                #     f"Оценка: {rating_text}\n"
                #     f"ID взаимодействия: {interaction_to_rate_id}\n"
                #     f"ID фрагмента: {item_id}"
                # )
                # await send_to_log_channel(context, log_rating_message, parse_mode="HTML")

                if success:
                    await query.edit_message_text(
                        text=query.message.text_html + "\n\n<i>Спасибо за вашу оценку!</i>",
                        parse_mode='HTML',
                        reply_markup=None
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
