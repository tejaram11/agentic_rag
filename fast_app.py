# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:31:31 2024

@author: TEJA
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json

import torch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter


from pdfloader import pdf_processor
from indexer_retriever import indexer
from embeddings import SentenceTransformerEmbeddings
from generator import LocalLLM
from semantics import SemanticSimilarityProcessor
from functions import docstring, get_special_today, extreme_fun_activity


app = FastAPI()

# CORS Middleware for handling cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



UPLOAD_FOLDER=os.path.join(os.getcwd(),"uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
embedding_model_path = "sentence-transformers/all-mpnet-base-v2"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

template = """
Assume you are school kid trying to answer the question given below.
you have studied about the topic present in the context below.
answer the question in 2 or 3 sentences only. If you can't
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

model = LocalLLM(model_path)
processor = pdf_processor(margin=100)
embeddings = SentenceTransformerEmbeddings(embedding_model_path, device)
indexes = indexer(chunk_size=512, overlap=50)

decision_maker=SemanticSimilarityProcessor(embedding_model=embeddings)
prompt = PromptTemplate.from_template(template)
parser = StrOutputParser()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Process the uploaded file and index it
    text = processor.process_pdf(file_path, skip_pages=[10], continous_pages=[13, 12, 11])
    indexes.chunk_and_index(text, embeddings)
    
    return JSONResponse(content={"message": "File uploaded and indexed successfully"}, status_code=200)


@app.post("/ask")
async def ask_question(question: str = Body(..., embed=True)):
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    
    chain = (
        {
            "context": itemgetter("question") | indexes.retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | parser
    )

    res = chain.invoke({"question": question})
    return JSONResponse(content={"answer": res}, status_code=200)


@app.post("/agent")
async def ask_agent(query: str = Body(..., embed=True)):
    if not query:
        raise HTTPException(status_code=400, detail="No question provided")
    
    #semantic similarity to decide whether to use the knowledge or not 
    decision_maker.embed_summary(processor.summary)
    score=decision_maker.calculate_similarity(query)
    if score>decision_maker.threshold:
        res = await ask_question(question=query)
        return res
    else:
        prompt=f"""
        you are an intelligent assistant that can decide whether we have to use provided tools or not for answering the provided query based on the similarity between query and function docstrings. 
        if we need to use internal tool say yes else say no. 
        say only a single word. 
        docstrings : {docstring}
        query: {query}

        output:
        """
        
        decision=model._call(prompt)
        decision=decision.strip()
        
        if decision=="yes":
            res=model.tool_call(query)
        else:
            res=model._call(query)
            
    return JSONResponse(content={"answer": res}, status_code=200)


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=5000)

 