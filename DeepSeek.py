import asyncio
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 加载 .env 文件中的环境变量

class DeepSeek():

    def __init__(self, model: str, tools = [], prompt: str = "", context: str = ""):
        self.deepseek = OpenAI(api_key = os.getenv("DS_API_KEY"), base_url = os.getenv("DS_BASE_URL"))
        self.model = model
        self.tools = tools
        self.messages = []
        if prompt:
             self.messages.append({"role": "system", "content": prompt})
        if context:
             self.messages.append({"role": "user", "content": context})

    async def chat(self, prompt: str):
        if prompt:
             self.messages.append({"role": "user", "content": prompt})

        response = self.deepseek.chat.completions.create(
            model = self.model,
            messages = self.messages,
            tools = self.getTools(),
            stream = True
        )

        content = ""
        toolCalls = []

        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content if delta.content else ""
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

if __name__ == "__main__":
    deepseek = DeepSeek("deepseek-chat")
    res = asyncio.run(deepseek.chat("Hello"))
    print(res)