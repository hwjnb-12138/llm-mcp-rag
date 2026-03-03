import asyncio
import os
import dotenv
import requests
from VectorStore import VectorStore, VectorStoreItem

dotenv.load_dotenv()

class Embedding():
    def __init__(self, model: str):
        self.model = model
        self.vectorStore = VectorStore()
    
    async def getEmbedding(self, document: str):
        payload = {
            "model": self.model,
            "input": document
        }
        headers = {
            "Authorization": f"Bearer {os.getenv("EMBEDDING_API_KEY")}",
            "Content-Type": "application/json"
        }

        response = requests.post(os.getenv("EMBEDDING_BASE_URL"), json=payload, headers=headers)
        data = response.json()
        return data["data"][0]["embedding"]
    
    async def getQueryEmbedding(self, query: str):
        return await self.getEmbedding(query)
    
    async def getDocumentEmbedding(self, document: str):
        res = await self.getEmbedding(document)
        await self.vectorStore.add(VectorStoreItem(res, document))
        return res
    
    async def search(self, query: str, top_k: int = 3):
        queryEmbedding = await self.getQueryEmbedding(query)
        return await self.vectorStore.search(queryEmbedding, top_k)

async def example():
    embedding = Embedding("BAAI/bge-m3")
    return await embedding.getQueryEmbedding("你好!")

if __name__ == "__main__":
    print(asyncio.run(example()))
