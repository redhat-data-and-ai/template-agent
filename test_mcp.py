import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    config = {
        "example-mcp-server": {
            "url": "http://localhost:5001/mcp/",
            "transport": "streamable_http",
        }
    }

    client = MultiServerMCPClient(config)
    tools = await client.get_tools()
    print("Available tools:")
    for t in tools:
        print(f"- {t.name}: {t.description}")

if __name__ == "__main__":
    asyncio.run(main())
