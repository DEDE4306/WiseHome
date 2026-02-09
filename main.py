import asyncio
from core.loop import loop

async def main():
    await loop()

if __name__ == "__main__":
    asyncio.run(main())