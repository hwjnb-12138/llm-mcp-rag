import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()  # 加载 .env 文件中的环境变量

class DeepSeek():

    def __init__(self, model: str, tools = [], prompt: str = "", context: str = ""):
        self.client = AsyncOpenAI(api_key = os.getenv("DS_API_KEY"), base_url = os.getenv("DS_BASE_URL"))
        self.model = model
        self.tools = tools
        self.messages = []
        
        # Combine system prompt and context into one system message for better compatibility
        system_content = prompt if prompt else "You are a helpful assistant."
        if context:
            context_str = "\n".join(context) if isinstance(context, list) else context
            system_content += f"\n\nContext information:\n{context_str}"
        
        self.messages.append({"role": "system", "content": system_content})

    async def chat(self, prompt: str):
        if prompt:
             self.messages.append({"role": "user", "content": prompt})

        # Prepare arguments for create()
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "stream": True
        }
        
        # Only add tools if they are actually provided
        api_tools = self.getTools()
        if api_tools:
            kwargs["tools"] = api_tools

        response = await self.client.chat.completions.create(**kwargs)

        content = ""
        toolCalls = []

        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
            if delta.tool_calls:
                for toolCall in delta.tool_calls:
                    if toolCall.index >= len(toolCalls):
                        toolCalls.append({
                            "id": "",
                            "function": {
                                "name": "",
                                "arguments": ""
                            }
                        })
                    currentCall = toolCalls[toolCall.index]
                    if toolCall.id:
                        currentCall["id"] += toolCall.id
                    if toolCall.function.name:
                        currentCall["function"]["name"] += toolCall.function.name
                    if toolCall.function.arguments:
                        currentCall["function"]["arguments"] += toolCall.function.arguments

        self.messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"]
                    }
                } for call in toolCalls if call["id"]
            ]
        })

        return {
            "content": content,
            "toolCalls": toolCalls
        }
    

    def getTools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in self.tools
        ]
    
    def appendToolResult(self, toolCallId: str, toolResult: str):
        self.messages.append({
            "role": "tool",
            "tool_call_id": toolCallId,
            "content": toolResult
        })
    
    async def close(self):
        await self.client.close()

if __name__ == "__main__":
    async def main():
        deepseek = DeepSeek("deepseek-chat")
        res = await deepseek.chat("Hello")
        print(res)
    
    asyncio.run(main())