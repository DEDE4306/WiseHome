import asyncio
from core.loop import loop
from config.constants import REACT_OUTPUT, USING_SPEECH_REC
import logging

logging.getLogger("funasr").setLevel(logging.ERROR)
logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)

async def main():
    await loop(react_output = REACT_OUTPUT, using_speech = USING_SPEECH_REC)

if __name__ == "__main__":
    asyncio.run(main())