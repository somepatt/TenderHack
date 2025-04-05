import logging
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, constants
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
)
import telegram.helpers

# Импортируем функции из нашего AI модуля
import ai_pipeline
import database

# --- Конфигурация Бота ---
TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN")
USE_LLM_GENERATION = os.environ.get(
    'USE_LLM_GENERATION', 'True').lower() == 'true'
# Можно добавить флаг для включения/отключения классификации
USE_LLM_CLASSIFICATION = os.environ.get(
    'USE_LLM_CLASSIFICATION', 'True').lower() == 'true'

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

    query_category = "Другое"  # Категория по умолчанию
    retrieval_ok, generation_ok = ai_pipeline.get_ai_status()

    if USE_LLM_CLASSIFICATION and generation_ok:
        # Отправляем индикатор печати, т.к. классификация может занять время
        await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
        query_category = ai_pipeline.classify_query_type_with_llm(user_query)
    elif not generation_ok:
        logger.warning(
            "Классификация LLM отключена, т.к. генеративная модель не инициализирована.")
        # Если генерации нет, все запросы пойдут по пути поиска в БЗ или "не найдено"
        query_category = "Общие вопросы"  # Предполагаем, что это вопрос к БЗ

    # --- 2. Логируем ЗАПРОС пользователя С КАТЕГОРИЕЙ ---
    request_interaction_id = database.log_interaction(
        user_telegram_id=user.id,
        is_from_user=True,
        message_text=user_query,
        query_category=query_category  # Записываем категорию
    )

    # --- 3. Выбираем стратегию ответа ---
    final_response_text = ""
    response_interaction_id = None
    kb_id_for_log = None
    similarity_for_log = None
    reply_markup = None

    # Показываем индикатор печати для основной обработки
    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

    # --- Стратегия А: Поиск в Базе Знаний ---
    if query_category in ai_pipeline.SEARCH_KB_CATEGORIES:
        logger.info(f"Категория '{query_category}' требует поиска в БЗ.")
        if not retrieval_ok:
            logger.error(
                "Retrieval Core не инициализирован. Ответ по БЗ невозможен.")
            final_response_text = "Извините, сервис временно недоступен (ошибка поиска). Попробуйте позже."
        else:
            context_list = ai_pipeline.retrieve_context(user_query)

            if context_list:
                best_context = context_list[0]
                kb_id_for_log = best_context.get('id')
                similarity_for_log = best_context.get('similarity')

                # Попытка генерации RAG ответа
                generated_answer = None
                if USE_LLM_GENERATION and generation_ok:
                    generated_answer = ai_pipeline.generate_answer_with_llm(
                        user_query, context_list)

                if generated_answer:
                    final_response_text = generated_answer
                else:
                    # Fallback на ответ из БЗ
                    logger.info(
                        "Ответ сформирован на основе найденного контекста (Fallback).")
                    response_parts = [f"Нашел похожий вопрос (схожесть: {best_context['similarity']:.2f}):",
                                      "<b>Ответ из базы знаний:</b>",
                                      f"<blockquote>{telegram.helpers.escape_html(best_context['answer'])}</blockquote>",
                                      f"<b>Источник:</b> {best_context.get('source', 'База знаний')}"]
                    final_response_text = "\n".join(response_parts)
            else:
                # Контекст не найден, но категория предполагала поиск
                logger.info(
                    "Релевантный контекст не найден для категории, требующей поиска.")
                # Опция 1: Просто сказать "не найдено"
                # final_response_text = "К сожалению, я не смог найти ответ на ваш вопрос в базе знаний. Попробуйте переформулировать."
                # Опция 2: Попробовать сгенерировать ответ LLM "из головы" (если она есть)
                if USE_LLM_GENERATION and generation_ok:
                    logger.info(
                        "Контекст не найден, пытаемся сгенерировать 'живой' ответ...")
                    # Используем generate_live_response, но с другой инструкцией
                    # TODO: Возможно, нужен отдельный промпт для этого случая
                    final_response_text = ai_pipeline.generate_live_response_with_llm(
                        user_query, "Другое")
                    if not final_response_text:  # Если и тут ошибка
                        final_response_text = "К сожалению, не могу найти информацию по вашему запросу и возникла ошибка при генерации ответа."
                    else:
                        final_response_text += "\n\n_(Ответ сгенерирован без использования базы знаний)_"
                else:
                    final_response_text = "К сожалению, я не смог найти ответ на ваш вопрос в базе знаний. Попробуйте переформулировать."

    # --- Стратегия Б: "Живое" общение ---
    else:
        logger.info(f"Категория '{query_category}' требует 'живого' ответа.")
        if USE_LLM_GENERATION and generation_ok:
            generated_live_answer = ai_pipeline.generate_live_response_with_llm(
                user_query, query_category)
            if generated_live_answer:
                final_response_text = generated_live_answer
            else:
                final_response_text = "Спасибо за ваше сообщение. Возникла ошибка при обработке."
        else:
            # Fallback, если LLM недоступна для живого ответа
            logger.warning("LLM недоступна для генерации 'живого' ответа.")
            if query_category == "Жалобы":
                final_response_text = "Приносим извинения за возможные неудобства. Ваша жалоба будет передана специалистам. Для срочных вопросов, пожалуйста, свяжитесь с поддержкой."
            elif query_category == "Обратная связь":
                final_response_text = "Спасибо за ваше мнение! Мы ценим вашу обратную связь."
            else:  # Приветствие, Прощание, Другое
                final_response_text = "Спасибо за ваше сообщение!"

    # --- 4. Логирование ОТВЕТА бота ---
    response_interaction_id = database.log_interaction(
        user_telegram_id=user.id,
        is_from_user=False,
        message_text=final_response_text,
        query_category=None,  # Категория для ответа не нужна
        request_interaction_id=request_interaction_id,
        matched_kb_id=kb_id_for_log,
        similarity_score=similarity_for_log
    )

    # --- 5. Формирование кнопок и отправка ---
    if response_interaction_id and kb_id_for_log:  # Оценка только для ответов по БЗ
        keyboard = [[
            InlineKeyboardButton(
                "👍", callback_data=f"rate_up_{response_interaction_id}"),
            InlineKeyboardButton(
                "👎", callback_data=f"rate_down_{response_interaction_id}"),
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
    elif query_category == "Жалобы" or (query_category in ai_pipeline.SEARCH_KB_CATEGORIES and not context_list):
        # Предлагаем оператора для жалоб или если ничего не нашли в БЗ
        keyboard = [[InlineKeyboardButton(
            "❓ Задать вопрос оператору", url="https://t.me/YOUR_SUPPORT_CONTACT")]]  # Замените
        reply_markup = InlineKeyboardMarkup(keyboard)
    # В остальных случаях кнопок нет (reply_markup = None)

    try:
        await update.message.reply_html(final_response_text, reply_markup=reply_markup)
    except Exception as e:
        logger.error(
            f"Ошибка отправки сообщения пользователю {user.id}: {e}", exc_info=True)
        fallback_text = final_response_text[:constants.MessageLimit.MAX_TEXT_LENGTH - 20] + "... (ответ урезан)" if len(
            final_response_text) > constants.MessageLimit.MAX_TEXT_LENGTH else "Произошла ошибка при отправке ответа."
        await update.message.reply_html(fallback_text)


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
    rate_type = parts[1]
    # ID теперь относится к interaction_id ответа бота
    interaction_to_rate_id = int(parts[3])  # Преобразуем в int

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
                # Редактируем сообщение (убираем кнопки или добавляем текст)
                await query.edit_message_text(
                    text=query.message.text_html + "\n\n<i>Спасибо за вашу оценку!</i>",
                    parse_mode='HTML',
                    reply_markup=None  # Убираем клавиатуру
                )
            else:
                await query.answer("Не удалось сохранить оценку.", show_alert=True)
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
        safe_message = telegram.helpers.escape_markdown(
            record['message_text'], version=2)  # Или escape_html
        # Используем Markdown V2 для совместимости с escape_markdown
        response_text += f"`{timestamp_str}`\n{prefix} {safe_message}\n\n"
        # Лимит Telegram на длину сообщения - около 4096 символов.
        if len(response_text) > 3800:  # Оставляем запас
            await update.message.reply_markdown_v2(response_text)
            response_text = ""  # Начинаем новое сообщение

    if response_text:  # Отправляем остаток, если есть
        await update.message.reply_markdown_v2(response_text)


# --- Основная функция запуска ---

def main() -> None:
    """Главная функция для инициализации и запуска бота."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical(
            "TELEGRAM_BOT_TOKEN не установлен! Бот не может быть запущен.")
        return

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
