import json
import os
import asyncio
from MCPClient import MCPClient
from DeepSeek import DeepSeek

class Agent():
    def __init__(self, model: str, mcpClients: list[MCPClient], sysPrompt: str = "", context: str = ""):
        self.model = model
        self.mcpClients = mcpClients
        self.sysPrompt = sysPrompt
        self.context = context
        self.deepSeek = None
    
    async def init(self):
        for mcp in self.mcpClients:
            await mcp.init()
        
        tools = []
        for mcp in self.mcpClients:
            tools.extend(mcp.tools)
        
        self.deepSeek = DeepSeek(self.model, tools, self.sysPrompt, self.context)
    
    async def close(self):
        for client in self.mcpClients:
            await client.cleanup()
        if self.deepSeek:
            await self.deepSeek.close()
    
    async def invoke(self, prompt: str):
        if not self.deepSeek:
            raise Exception("Agent not initialized")
        
        # Ensure context is handled correctly in DeepSeek initialization if needed
        # But here we just proceed with the chat
        res = await self.deepSeek.chat(prompt)

        while True:
            # 处理工具调用
            if res["toolCalls"]:
                for toolCall in res["toolCalls"]:
                    # 查询处理该工具的MCP
                    mcp = next((client for client in self.mcpClients
                                if any(tool.name == toolCall["function"]["name"] for tool in client.tools)), None)
                    if mcp:
                        print("Calling tool: " + toolCall["function"]["name"])
                        toolResult = await mcp.callTool(toolCall["function"]["name"], json.loads(toolCall["function"]["arguments"]))
                        print("ToolResult: " + str(toolResult))
                        self.deepSeek.appendToolResult(toolCall["id"], json.dumps(str(toolResult)))
                    else:
                        self.deepSeek.appendToolResult(toolCall["id"], "Tool Not Found")
                res = await self.deepSeek.chat("")
                continue
            
            return res["content"]

async def example():
    currentDirectory = os.getcwd()
    fetchMCP = MCPClient("fetch", "uvx", ["mcp-server-fetch"])
    fileMCP = MCPClient("file", "npx", ["-y",
        "@modelcontextprotocol/server-filesystem",
        currentDirectory])
    agent = Agent("deepseek-chat", [fetchMCP, fileMCP])
    await agent.init()
    res = await agent.invoke(f"爬取 https://news.ycombinator.com 的内容, 并且总结后保存在 {currentDirectory} 目录下的news.md文件中")
    print(res)
    await agent.close()

if __name__ == "__main__":
    asyncio.run(example())