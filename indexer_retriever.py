# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:33:26 2024

@author: TEJA
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

class indexer():
    def __init__(self,chunk_size=200,overlap=10):
        self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True
            )
        self.vectorstore = None
        self.retriever =  None

        
        
    def chunk_and_index(self,final_text,embedding_model):
        final_text=self.splitter.create_documents(final_text)
        docs=self.splitter.split_documents(final_text)
        self.vectorstore=FAISS.from_documents(docs, embedding_model)
        self.retriever=self.vectorstore.as_retriever()
    
        