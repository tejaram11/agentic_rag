# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:16:43 2024

@author: TEJA
"""

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str, device:str):
        self.model = SentenceTransformer(model_name,device=device)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.model.encode(documents)

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0]