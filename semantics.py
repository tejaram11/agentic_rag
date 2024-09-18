# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:55:21 2024

@author: TEJA
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSimilarityProcessor:
    def __init__(self, embedding_model,alpha=0.5):
        """
        Initialize the class with a sentence transformer embedding model.
        
        :param embedding_model: A SentenceTransformerEmbeddings instance
        """
        self.embedding_model = embedding_model
        self.summary_embedding = None  # Will hold the vectorized summary
        self.threshold= alpha

    def embed_summary(self, summary_text):
        """
        Embeds the summary text.
        
        :param summary_text: The summary text extracted from the document
        :return: The embedding of the summary
        """
        self.summary_embedding = self.embedding_model.embed_documents([summary_text])[0]
        return self.summary_embedding

    def calculate_similarity(self, query_text):
        """
        Embeds the query and calculates cosine similarity between the query and the summary.
        
        :param query_text: The query from the user
        :return: Cosine similarity between the query and the summary (float)
        """
        query_embedding = self.embedding_model.embed_query(query_text)
        
        # Reshape vectors for cosine similarity calculation
        summary_vec = np.array(self.summary_embedding).reshape(1, -1)
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(summary_vec, query_vec)[0][0]
        
        return similarity