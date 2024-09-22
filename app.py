# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:31:31 2024

@author: TEJA
"""
import argparse
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import requests
import base64
import io


import torch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter


from pdfloader import pdf_processor
from indexer_retriever import indexer
from embeddings import SentenceTransformerEmbeddings
from generator import LocalLLM
from semantics import SemanticSimilarityProcessor
from functions import tools, docstring


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
    #res = res[len(prompt):]
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
        decision=decision[len(prompt):len(prompt)+4]
        decision=decision.strip()
        
        if decision=="yes":
            res=model.tool_call(query,tools)
        else:
            res=model._call(query)
            
    return JSONResponse(content={"answer": res}, status_code=200)


@app.post("/generate-audio")
async def generate_audio(request: Request):
    print(request)
    data = await request.json()
    print(data)
    URL = "https://api.sarvam.ai/text-to-speech"
    key= "e0d456d9-5d0d-4e45-ae0c-92c1db82b29a"
    text = data.get('text')
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided for conversion.")

    payload = {
        "inputs": [text],
        "target_language_code": "te-IN",
        "speaker": "meera",
        "pitch": 0,
        "pace": 1.65,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }

    headers = {"Content-Type": "application/json",
               "api-subscription-key": key }

    try:
        
        response = requests.post(URL, json=payload, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        tts_response = response.json()

        if "audios" not in tts_response or not tts_response["audios"]:
            raise HTTPException(status_code=500, detail="No audio data received from TTS API.")
            
        audio_base64 = tts_response["audios"][0]
        audio_bytes = base64.b64decode(audio_base64)
        audio_file_path = "output_audio.wav"
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(audio_bytes)
        audio_stream = io.BytesIO(audio_bytes)
        return StreamingResponse(audio_stream, media_type="audio/wav")

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error in TTS API call: {str(e)}")

def setup_remote_server():
    import nest_asyncio
    from pyngrok import ngrok

    auth_token = "2mD79q8xYOrgmoWpQiq6jBY6az4_5Tx5ic5BTZEyqqrgK4ts3"
    ngrok.set_auth_token(auth_token)
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()


if __name__ == "__main__":
   args_parser = argparse.ArgumentParser(description="Run FastAPI server")
   args_parser.add_argument("--use-remote", action="store_true", help="Run server with ngrok")
   
   args = args_parser.parse_args()
   
   if args.use_remote:
       setup_remote_server()
       uvicorn.run(app, host="0.0.0.0", port=8000)
   else:
       uvicorn.run(app, host="0.0.0.0", port=5000)

