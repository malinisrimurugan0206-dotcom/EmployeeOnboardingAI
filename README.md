Employee Onboarding AI Assistant

Project Overview

The Employee Onboarding AI Assistant is an AI-powered application that helps employees find information from HR policy documents. Users can upload a PDF and ask questions about its content using a simple chat interface.

Objectives

- Simplify access to employee onboarding information.
- Answer questions using uploaded HR policy documents.
- Maintain conversation history for multiple questions.
- Avoid making up policy details when information is not available.

Technologies Used

- Python
- Streamlit
- LangChain
- Ollama
- Llama 3.2
- ChromaDB
- PDF processing with PyPDF
- Nomic Embed Text for embeddings

Key Features

- Upload HR policy PDF documents.
- Extract and split PDF text into smaller chunks.
- Retrieve relevant information using vector similarity search.
- Generate answers using a locally running Llama 3.2 model.
- Display multiple questions and answers in one chat session.
- Support general questions when no PDF is uploaded.
- Direct users to HR when policy information is not found in the uploaded document.

How It Works

1. The user uploads an HR policy PDF.
2. The application extracts text from the PDF.
3. The text is split into smaller chunks.
4. ChromaDB stores the chunks as searchable vector data.
5. When a question is asked, relevant chunks are retrieved.
6. The language model generates an answer based on the retrieved context.

Installation and Setup

Prerequisites

- Python installed
- Ollama installed and running
- Llama 3.2 and Nomic Embed Text models downloaded

Install Dependencies

py -m pip install streamlit langchain-text-splitters langchain-ollama langchain-chroma pypdf

Download Required Models

ollama pull llama3.2
ollama pull nomic-embed-text

Run the Application

py -m streamlit run app.py

Open the local URL shown in the terminal to use the application.

Limitations

- Answers depend on the text that can be extracted from the uploaded PDF.
- Scanned PDFs may require OCR before their text can be processed.
- The application is a prototype and should not replace official HR guidance.
- General questions without an uploaded PDF are answered by the language model and may not reflect company policies.

Conclusion

The Employee Onboarding AI Assistant demonstrates how Retrieval-Augmented Generation (RAG), embeddings, vector search, and a local language model can help employees find information from HR policy documents.