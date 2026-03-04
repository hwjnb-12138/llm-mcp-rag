from typing import Any
import asyncio
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, name: str, command: str, args: list[str]):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.name = name
        self.command = command
        self.args = args
        self.tools = []

    async def callTool(self, name: str, args: dict[str, Any]):
        if not self.session:
            raise Exception("Session not initialized")
        return await self.session.call_tool(name, args)
    
    @asynccontextmanager
    async def server_session(self):
        """Connect to an MCP server and maintain session context."""
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=None
        )
        
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # List available tools
                response = await session.list_tools()
                self.tools = response.tools
                self.session = session
                print(f"\\nConnected to {self.name} server with tools:", [tool.name for tool in self.tools])
                
                try:
                    yield
                finally:
                    self.session = None
                    print(f"Closed connection to {self.name}")

async def example():
    mcp = MCPClient("fetch", "uvx", ["mcp-server-fetch"])
    async with mcp.server_session():
        print(mcp.tools)

if __name__ == "__main__":
    asyncio.run(example())