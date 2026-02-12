import asyncio
from core.loop import loop
from config.constants import REACT_OUTPUT

async def main():
    await loop(react_output = REACT_OUTPUT)

if __name__ == "__main__":
    asyncio.run(main())