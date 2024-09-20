import os
import json
import requests
from urllib.parse import urlparse
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI
from tiktoken import encoding_for_model

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


chroma_client = chromadb.PersistentClient(path="./chroma_db")

default_ef = embedding_functions.DefaultEmbeddingFunction()

def get_repo_info(repo_url):
    """
    Extract the owner and repository name from the GitHub URL.
    """
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub repository URL.")
    owner = path_parts[0]
    repo = path_parts[1].replace('.git', '')
    return owner, repo

def get_contents(owner, repo, path=""):
    """
    Recursively retrieve the contents of the repository, excluding image and media files.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": "main"}  # You can change the branch if needed
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    items = response.json()
    contents = []

    if not isinstance(items, list):
        items = [items]

    # Exclude image and media formats
    excluded_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.mp4', '.mp3', '.wav', '.avi', '.mov']

    for item in items:
        if item['type'] == 'file' and any(item['path'].endswith(ext) for ext in excluded_extensions):
            continue

        if item['type'] == 'file':
            file_content = get_file_content(item['download_url'])
            contents.append({
                'type': 'file',
                'path': item['path'],
                'content': file_content
            })
        elif item['type'] == 'dir':
            dir_contents = get_contents(owner, repo, item['path'])
            contents.append({
                'type': 'dir',
                'path': item['path'],
                'contents': dir_contents
            })
    return contents


def get_file_content(download_url):
    """
    Retrieve the raw content of a file.
    """
    response = requests.get(download_url)
    response.raise_for_status()
    return response.text


def flatten_contents(contents):
    """
    Flatten the nested contents into a list of files with paths and contents.
    """
    flat_files = []

    def _flatten(items):
        for item in items:
            if item['type'] == 'file':
                flat_files.append({
                    'path': item['path'],
                    'content': item['content']
                })
            elif item['type'] == 'dir':
                _flatten(item['contents'])

    _flatten(contents)
    return flat_files


def ingest_into_vector_db(files, collection_name):
    """
    Ingest files into the vector database using embeddings.
    """
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=default_ef)

    texts = [file['content'] for file in files]
    metadatas = [{'path': file['path']} for file in files]
    ids = [file['path'] for file in files]  # Using file paths as IDs

    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    print(f"Ingested {len(files)} documents into the vector database.")


def query_vector_db(query, collection_name, top_k=5):
    """
    Query the vector database to retrieve the top_k most relevant documents.
    """
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=default_ef)
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    # results['documents'] is a list of lists
    # Flatten it to a single list
    retrieved_docs = []
    for doc_list in results['documents']:
        for doc in doc_list:
            retrieved_docs.append(doc)
    return retrieved_docs


def truncate_to_token_limit(text, max_tokens, encoding='gpt-4'):
    """
    Truncate the text to fit within the specified token limit.
    """
    tokenizer = encoding_for_model(encoding)
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)


def summarize_text(text, max_tokens=200):
    """
    Summarize the text to fit within a limited number of tokens.
    """
    prompt = f"Summarize the following text to {max_tokens} tokens:\n\n{text}\n\nSummary:"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


def generate_gpt4_response(query, context_docs, max_context_tokens=6000):
    """
    Generate a response using OpenAI's GPT-4 based on the query and retrieved context documents.
    """
    # Summarize to reduce token size
    summarized_contexts = [summarize_text(doc, max_tokens=200) for doc in context_docs]

    # Combine into a single string
    combined_context = "\n\n".join(summarized_contexts)
    
    # Truncate
    context = truncate_to_token_limit(combined_context, max_context_tokens)

    prompt = f"""You are an assistant that provides detailed answers based on the following context:


Context:
{context}

Question:
{query}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error generating GPT-4 response: {e}")
        return "I'm sorry, I couldn't process your request at the moment."


def query_main():
    """
    Main function for querying the vector database.
    """
    collection_name = input("Enter the collection name to query: ").strip()
    query = input("Enter your query: ").strip()
    top_k = input("Enter the number of top documents to retrieve (default 5): ").strip()
    if not top_k.isdigit():
        top_k = 5
    else:
        top_k = int(top_k)

    print("Retrieving relevant documents from the vector database...")
    retrieved_docs = query_vector_db(query, collection_name, top_k=top_k)
    print(f"Retrieved {len(retrieved_docs)} documents.")

    # Extract the content from the retrieved documents
    context_docs = [doc for doc in retrieved_docs]

    print("Generating response using GPT-4...")
    answer = generate_gpt4_response(query, context_docs)
    print("\n=== GPT-4 Response ===")
    print(answer)
    print("======================\n")


def main():
    """
    Main function to either ingest data into the vector database or query it.
    """
    while True:
        print("Choose an option:")
        print("1. Ingest GitHub repository into vector database")
        print("2. Query the vector database")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == '1':
            repo_url = input("Enter the GitHub repository URL: ").strip()
            try:
                owner, repo = get_repo_info(repo_url)
                print(f"Processing repository '{owner}/{repo}'...")
                repo_contents = get_contents(owner, repo)

                output_filename = f"{repo}_structured_content.json"
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(repo_contents, f, indent=2)

                print(f"Repository contents have been saved to '{output_filename}'.")

                flat_files = flatten_contents(repo_contents)
                ingest_into_vector_db(flat_files, collection_name=f"{owner}_{repo}_collection")

            except Exception as e:
                print(f"An error occurred: {e}")

        elif choice == '2':
            try:
                query_main()
            except Exception as e:
                print(f"An error occurred during querying: {e}")

        elif choice == '3':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
