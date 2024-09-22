# Agentic RAG System
### Introduction

This project is part of an assignment to build a Retrieval-Augmented Generation (RAG) system that leverages PDFs to answer user queries. The project involves two main tasks:

1. Building a RAG system using NCERT PDFs and serving it through a FastAPI endpoint.
2. Extending the system by introducing an agent that can decide when to query the VectorDB or perform other actions based on the user's query.

The system provides a simple web-based user interface to upload PDFs, ask questions, and interact with the agent.

## Assignment Requirements
### Part 1: Building a RAG System
    Use NCERT PDF text as the data source for domain knowledge
    Chunk and index the PDF content in a VectorDB.
    Use a FastAPI endpoint to serve the RAG system and process user queries.
### Part 2: Building an Agent
    The agent decides when to call the VectorDB based on the user's query.
    The agent can decide the tool to use based on user query and execute it.
### Project Structure


    sarvam_assignment/	
    │
    ├── Outputs /                    # Folder for images 
    ├── Uploads /                    # Folder for saving uploaded files
    ├── app.py                       # Main FastAPI application file
    ├── embeddings.py                # Handles embedding creation using Sentence Transformers
    ├── functions.py                 # Tools for  Agent
    ├── generator.py                 # Logic for generating responses using RAG and agent logic
    ├── indexer_retriever.py         # Indexing PDF text in FAISS and retrieving relevant documents
    ├── pdfloader.py                 # PDF loader to extract and process text from NCERT PDFs  
    ├── semantics.py                 # Module to handle semantic similarity processing
    ├── webpage.html                 # Simple web interface to upload PDFs and interact with the system  
    ├── requirements.txt             # Project  dependencies
    └── README.md                    # Documentation (This file)


### Features
- PDF Processing: Upload PDFs, extract text, and chunk it for indexing.
- RAG System: Queries are processed using FAISS for retrieval and the LLM for response generation.
- Agent: Smart agent decides whether to call the VectorDB based on semantic similarity between the user's query and the document summary. If the query is irrelevant, it avoids querying the database.
- Web Interface: A frontend where users can:
    - Upload PDFs in runtime rather than static pdf usage for vectorDB
    - Ask questions directly to RAG
    - Ask Agent and use tools
    - Listen to the Answer
    
## Flowchart
Assignment - 1: 
![RAG](outputs/RAG.jpg)

Assignment - 2 & 3:
![RAG_agent](outputs/rag_result.jpg)
## Setup Instructions

Clone the repository:
    git clone https://github.com/your-repo/sarvam_assignment.git
    cd sarvam_assigment

Install the required dependencies:

    pip install -r requirements.txt

Running the Application

- Start the FastAPI server:
    - if you want to compute in remote server
    python app.py --use-remote

    - Then the fastAPI app will create a tunnel using ngrok and provide a PUBLIC_URL
    - Replace this PUBLIC_URL in API_URL in webpage using "change API URL" 


    - if you want to compute in same machine
    python app.py


- Open "Webpage.html" in your browser


## Endpoints
- /upload:
Upload PDF: Allows uploading an NCERT PDF and processing it into chunks for indexing.
- /ask :
Ask a Question: Submits a question to the system and retrieves an answer by querying the VectorDB and generating a response.
- /agent:
Ask Agent: Submits a query to the agent, which decides whether to call the VectorDB or use the tool or directly generate the response. 
- /generate-audio:
Generate Audio: Generates audio for final answer using sarvam API


## Sample results:
- webpage:
![sample](outputs/webpage_sample.jpg)

- Changing API url: 
![api_change](outputs/api_change.jpg)

- Using RAG:
![RAG_result](outputs/rag_result.jpg)

- Using Agent:
![Agent](outputs/agent_result.jpg)

- Using Tool:
![Tool](outputs/tool_result.jpg)

- Agent Voice:
