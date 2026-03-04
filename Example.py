import os
import asyncio
from Agent import Agent
from MCPClient import MCPClient
from Embedding import Embedding


currentDir = os.getcwd()
fetchMCP = MCPClient("fetch", "uvx", ["mcp-server-fetch"])
fileMCP = MCPClient("file", "npx", ["-y",
    "@modelcontextprotocol/server-filesystem",
    currentDir])


#prompt = f"""从 knowledge 目录下找到Emily和James的信息, 总结后保存在{currentDir}/summary.md文件中,
#根据总结的信息创造一个关于他们的故事, 保存在{currentDir}/story.md文件中"""
prompt = """请介绍一下你自己"""

async def example():
    embedding = Embedding("BAAI/bge-m3")
    knowledgeDir = os.path.join(currentDir, "knowledge")
    for root, _, files in os.walk(knowledgeDir):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                await embedding.getDocumentEmbedding(content)
    
    context = await embedding.search(prompt)
    agent = Agent("deepseek-chat", [fetchMCP, fileMCP], context=context)
    async with agent.session():
        res = await agent.invoke(prompt)
        print(res)

if __name__ == "__main__":
    asyncio.run(example())
