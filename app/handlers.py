import logging
import pickle
import asyncio
from aiogram import F, Router
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

router = Router()

# Render logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

def load_model():
    try:
        with open('ai_model/pkl/spam_classifier.pkl', 'rb') as model_file, open('ai_model/pkl/vectorizer.pkl', 'rb') as vectorizer_file:
            model = pickle.load(model_file)
            vectorizer = pickle.load(vectorizer_file)
        logging.info("Successfully loaded the spam classifier model and vectorizer.")
        return model, vectorizer
    except Exception as e:
        logging.error(f"Error loading model/vectorizer: {e}")
        raise

def predict_message(model, vectorizer, message):
    try:
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)
        return prediction[0]
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return "not_spam"  # Return "not_spam" if prediction fails

model, vectorizer = load_model()

deleted_messages = {}
delete_tasks = {}

def maintain_deleted_message_limit(limit=20):
    if len(deleted_messages) > limit:
        oldest_message_id = next(iter(deleted_messages))
        del deleted_messages[oldest_message_id]

@router.message(F.text)
async def handle_message(message: Message):
    global deleted_messages, delete_tasks

    user_id = message.from_user.id
    message_id = message.message_id
    message_text = message.text

    prediction = predict_message(model, vectorizer, message_text)

    if prediction == "spam":
        deleted_messages[message_id] = message

        maintain_deleted_message_limit(limit=20)

        try:
            await message.delete()
            logging.info(f"Deleted spam message | User ID: {user_id} | Message ID: {message_id} | Content: {message_text}")
        except Exception as e:
            logging.error(f"Failed to delete message ID {message_id}: {e}")

        inline_kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="Это был не спам", callback_data=f"restore_{message_id}")]
            ]
        )
        try:
            warning_message = await message.answer("Обнаружен спам, сообщение удалено.", reply_markup=inline_kb, disable_notification=True)

            delete_task = asyncio.create_task(delete_warning_after_timeout(warning_message, message_id))

            delete_tasks[message_id] = delete_task

        except Exception as e:
            logging.error(f"Failed to send notification for message ID {message_id}: {e}")



async def delete_warning_after_timeout(warning_message: Message, message_id: int):
    await asyncio.sleep(40)

    if message_id in deleted_messages:
        try:
            await warning_message.delete()
            logging.info(f"Deleted the spam notification for Message ID: {message_id}")
        except Exception as e:
            logging.error(f"Failed to delete spam notification for Message ID {message_id}: {e}")

        delete_tasks.pop(message_id, None)

@router.callback_query(F.data.startswith('restore_'))
async def restore_message(callback: CallbackQuery):
    global delete_tasks

    message_id_str = callback.data.split('_')[1]
    
    try:
        message_id = int(message_id_str)
    except ValueError:
        await callback.answer("Некорректный ID.", show_alert=True)
        logging.warning(f"Invalid message ID received in callback: {message_id_str}")
        return

    if message_id in deleted_messages:
        original_message = deleted_messages[message_id]
        original_user_id = original_message.from_user.id
        message_text = original_message.text
        current_user_id = callback.from_user.id

        if current_user_id == original_user_id:
            try:
                username = callback.from_user.first_name or callback.from_user.username
                await callback.message.answer(f"Восстановленное сообщение от {username}: {message_text}")
                
                logging.info(f"Restored message | User ID: {original_user_id} | Message ID: {message_id} | Content: {message_text}")

                await callback.message.delete()

                if message_id in delete_tasks:
                    delete_tasks[message_id].cancel()
                    delete_tasks.pop(message_id, None)

            except Exception as e:
                logging.error(f"Failed to restore message ID {message_id}: {e}")

            del deleted_messages[message_id]

            await callback.answer("Сообщение восстановлено.", show_alert=True)
        else:
            await callback.answer("Только отправитель может восстанавливать свое сообщение.", show_alert=True)
            logging.warning(f"User {current_user_id} attempted to restore message from user {original_user_id}")
    else:
        await callback.answer("Данное сообщение больше не может быть восстановлено.", show_alert=True)
        logging.warning(f"Attempted to restore non-existent message ID: {message_id}")
