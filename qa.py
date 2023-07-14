#!/usr/bin/env python3

# This is a Q&A bot for testing the Pinecone DB
# Prompt and Role should be written in the file "config.json"

import openai
import pinecone
import os
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from dotenv import load_dotenv
import json

# env path
load_dotenv('.env.local')

# Models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# Token Limit
LIMIT = 4096

# Ask user for the Pinecone index name
INDEX_NAME = input("Please enter the index name: ")

# Define the file path to config.json file
CONFIG_FILE_PATH = 'config.json'

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index = pinecone.Index(INDEX_NAME)


# search function using Pinecone
def strings_ranked_by_relatedness(
    query: str,
    top_n: int = 100
) -> object:
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    results = index.query(query_embedding, top_k=top_n, include_metadata=True)
    return results


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def load_from_config(file_path, key):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data[key]

# Pickup all items in metadata
def get_metadata(match):
    info = []
    for key, value in match["metadata"].items():
        info.append(f'{key}:\n{value}\n')
    return '\n'.join(info)


def query_message(
    query: str,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    results = strings_ranked_by_relatedness(query)
    introduction = load_from_config(CONFIG_FILE_PATH, 'introduction')
    question = f"\n\nQuestion: {query}"

    message = introduction + question

    for i, match in enumerate(results["matches"], start=1):
        metadata = get_metadata(match)

        # For display top 3 in order to check if Pinecone DB picked up correctly.
        if i < 4:
            if i == 1:
                print(f"Top 3 from Pinecone DB:")
            print(f"\n= Rank:{i} =\n{metadata}")

        next_article = f'\n\n\nPossible relevant articles {i}:\n\n\n{metadata}\n\n'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article

    print("Generating messages...")
    return message

def ask(
    query: str,
    model: str = GPT_MODEL,
    token_budget: int = LIMIT - 750,
    print_message: bool = False,
) -> str:
    message = query_message(query, model=model, token_budget=token_budget)

    if print_message:
        print(message)
    
    system_message = load_from_config(CONFIG_FILE_PATH, 'system')

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": message},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2
    )

    response_message = response["choices"][0]["message"]["content"]
    return response_message


while True:
    print("\n")
    message = input("Please input what you want to search in embeddings OR type 'exit' to quit: ")
    
    if message.lower() == 'exit':
        break

    print("\n")
    print(ask(message))
    print("\n")
