# GitHub Repository to Vector Database Ingestion and Query Tool

This script allows you to ingest the contents of a GitHub repository into a vector database using embeddings and query the database to retrieve relevant documents. It integrates with OpenAI's GPT-4 to provide detailed answers based on the query and the retrieved documents.

## Features

- Extracts contents from a GitHub repository, excluding image and media files.
- Ingests the structured content into a Chroma vector database.
- Queries the vector database to find the most relevant documents.
- Utilizes GPT-4 to generate responses based on the query and retrieved context documents.

## Requirements

- Python 3.11
- Make a virtual environment
- pip install -r requirements.txt
- Set up OpenAI API Key: Ensure that the OPENAI_API_KEY environment variable is set with your OpenAI API key.
- run main.py and follow the prompts


```bash

Usage

Choose an Option:
    - Ingest GitHub repository into vector database: Enter 1    

    - Provide the GitHub repository URL.

The script will process the repository, save the contents to a JSON file, and ingest the contents into the vector database.

    - Query the vector database: Enter 2

    - Provide the collection name and your query.

The script will retrieve relevant documents and use GPT-4 to generate a detailed response.