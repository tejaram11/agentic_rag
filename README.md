# PDF RAG System
### Introduction

This project is part of an assignment to build a Retrieval-Augmented Generation (RAG) system that leverages NCERT textbook PDFs to answer user queries. The project involves two main tasks:

    1. Building a RAG system using NCERT PDFs and serving it through a FastAPI endpoint.
    2. Extending the system by introducing an agent that can decide when to query the VectorDB or perform other actions based on the user's query.

The system provides a simple web-based user interface to upload PDFs, ask questions, and interact with the agent.

## Assignment Requirements
### Part 1: Building a RAG System
    Use NCERT PDF text as the data source.
    Chunk and index the PDF content in a VectorDB (FAISS).
    Use a FastAPI endpoint to serve the RAG system and process user queries.
### Part 2: Building an Agent
    The agent decides when to call the VectorDB based on the user's query.
    Introduce at least one more action/tool that the agent can invoke based on the query.
### Project Structure

sarvam_assignment/	
  │
  ├── app.py                       # Main FastAPI application file
  ├── embeddings.py                # Handles embedding creation using Sentence Transformers
  ├── generator.py                 # Logic for generating responses using RAG and agent logic
  ├── indexer_retriever.py         # Indexing PDF text in FAISS and retrieving relevant documents
  ├── pdfloader.py                 # PDF loader to extract and process text from NCERT PDFs  
  ├── semantics.py                 # Module to handle semantic similarity processing
  ├── webpage.html                 # Simple web interface to upload PDFs and interact with the system  
  ├── requirements.txt             # Python dependencies
  └── README.md                    # Documentation (This file)


### Features
    PDF Processing: Upload NCERT PDFs, extract text, and chunk it for indexing.
    RAG System: Queries are processed using FAISS for retrieval and the LLM for response generation.
    Agent: Smart agent decides whether to call the VectorDB based on semantic similarity between the user's query and the document summary. If the query is irrelevant, it avoids querying the database.
    Web Interface: A frontend where users can:
    Upload PDFs.
    Ask questions.
    Interact with the agent.
    
## Setup Instructions
Prerequisites
Ensure you have the following installed:
Python 3.8+
pip (Python package manager)
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo/sarvam_assignment.git
cd sarvam_assigment
Install the required dependencies:

bash
pip install -r requirements.txt
Running the Application
Start the FastAPI server:
bash
python app.py

This will start the server locally at http://127.0.0.1:8000.

Access the Web Interface:

Open webpage.html in a browser or serve it through any static server. You can also host it locally through python -m http.server.

Endpoints
Upload PDF: Allows uploading an NCERT PDF and processing it into chunks for indexing.
Ask a Question: Submits a question to the system and retrieves an answer by querying the VectorDB and generating a response.
Ask Agent: Submits a query to the agent, which decides whether to call the VectorDB or perform another action based on semantic similarity.
File Overview
1. app.py
The main FastAPI application file. It exposes endpoints to handle:

PDF uploads.
Question asking.
Agent queries.
2. pdfloader.py
Handles extracting text from NCERT PDFs, dividing it into chunks for efficient indexing, and storing it for retrieval.

3. indexer_retriever.py
Uses the FAISS VectorDB for indexing document chunks and retrieving relevant documents when a query is made.

4. semantics.py
Contains the logic for calculating semantic similarity between a query and the document summary using cosine similarity. This is used by the agent to determine whether to query the VectorDB.

5. embeddings.py
Handles embedding the text and query into vector representations using Sentence Transformers, which are necessary for similarity calculations and vector retrieval.

6. webpage.html
Provides a simple user interface to upload PDFs, ask questions, and interact with the agent. The frontend communicates with the FastAPI backend using JavaScript and Axios.

Usage
Uploading PDFs
Select an NCERT PDF file and click "Upload" to process the file into indexed chunks.
Asking a Question
Enter a question related to the NCERT content. The system will retrieve relevant chunks from the VectorDB and generate a response.
Asking the Agent
Enter a query. The agent decides whether to query the VectorDB based on the relevance of the query to the document's summary.
Future Enhancements
Additional Tools: Introduce more creative actions or tools for the agent, such as summarizing the document or providing definitions for terms found in the text.
Improved UI: Enhance the web interface for a more interactive and visually appealing experience.
Advanced Query Handling: Add more robust logic to the agent for handling complex queries and multiple actions.
