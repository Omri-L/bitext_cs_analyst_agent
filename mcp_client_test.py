import asyncio
from fastmcp import Client


async def main():
    client = Client("mcp_server.py")

    async with client:
        result = await client.call_tool("get_categories", {})
        print(result)

        result = await client.call_tool(
            "get_examples",
            {
                "n": 3,
                "category": "REFUND"
            }
        )

        print(result)


asyncio.run(main())