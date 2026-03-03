import math

class VectorStoreItem():
    def __init__(self, embedding: list[float], document: str):
        self.embedding = embedding
        self.document = document

class VectorStore():
    def __init__(self):
        self.vectorStore: list[VectorStoreItem] = []
    
    async def add(self, item: VectorStoreItem):
        self.vectorStore.append(item)
    
    async def search(self, query: list[float], top_k: int = 3):
        # Calculate scores synchronously since cosSimilarity is now sync
        result = sorted(self.vectorStore,
            key = lambda x: self.cosSimilarity(query, x.embedding),
            reverse = True
        )
        return [item.document for item in result[:top_k]]
        
    def cosSimilarity(self, query: list[float], document: list[float]):
        dotProduct = sum(a * b for a, b in zip(query, document))
        queryMagnitude = math.sqrt(sum(a * a for a in query))
        documentMagnitude = math.sqrt(sum(b * b for b in document))
        if queryMagnitude == 0 or documentMagnitude == 0:
            return 0
        return dotProduct / (queryMagnitude * documentMagnitude)
