import asyncio
from aiogram import Bot, Dispatcher
from app.handlers import router
import os
from keep_alive import keep_alive

keep_alive()

async def main():
    # bot = Bot(token=os.environ.get('TELEGRAM_BOT_TOKEN'))
    bot = Bot(token = '7920824189:AAEF6Y58VHQQzShKFI-b05w3lpMFA2Fz-kE')
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Bot stopped')
