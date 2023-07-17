#!/usr/bin/env python3

# This script is for auto importing data from a CSV file with multiple columns into a vector database.

import os
import pandas as pd
import openai
import pinecone
import csv
import json
import time
from dotenv import load_dotenv


# Load the .env.local file
load_dotenv('.env.local')
openai.api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")

# Define the embeddings model
EMBEDDING_MODEL = "text-embedding-ada-002"


def read_jsonl(filename_jsonl):
    try:
        with open(filename_jsonl, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
        return data
    except FileNotFoundError:
        return None

# For saving embeddings to a specified file.
# All data will be inserted in the target file. Not clear all before writing.
# clear_file: {True: Clears existing data and overwrites it, False: Appends new data after the existing data.}
def record_embeddings(data_set, embedding_column_name, clear_file, output_file_name = './data/temp.jsonl'):
    if clear_file:
        # Open the file in write mode to create it or clear it if it already exists
        with open(output_file_name, 'w', encoding='utf-8') as json_file:
            pass
    with open(output_file_name, 'a', encoding='utf-8') as json_file:
        print("\nStart embeddigns process for")
        counter = 1
        id_num = 1
        for row in data_set:
            
            # for avoiding API rate limit, wait 60sec every 5000
            if counter % 5000 == 0:
                print("Waiting...")
                time.sleep(60)
            
            # If an embeddings error is returned, then back to menu.
            try:
                embedding = openai.Embedding.create(input=row[embedding_column_name], model=EMBEDDING_MODEL)["data"][0]["embedding"]
            except Exception as e:
                print(f"Error when creating embeddings in {row}: {e}\n\nEmbedding creation failed. Please try agin.")
                return False
            
            print(f"id_{id_num}")
            row_dict = row.copy()
            row_dict["embeddings"] = embedding
            
            json.dump(row_dict, json_file, ensure_ascii=False)
            json_file.write("\n")

            counter += 1
            id_num += 1

# For saving Vector and meta data in online vectorDB(s)
def vectordb(filename_json = './data/temp.jsonl'):
    # For Pinecone
    index_name = input("\nEnter the Pinecone Index name: ")
    dimension = 1536

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    index = pinecone.Index(index_name)

    # Create index if not exists
    if index_name not in pinecone.list_indexes():
        print("\nStart creating the Index. It takes some time. (at least 3min or more...)")
        try:
            pinecone.create_index(index_name, dimension, metric="cosine", pod_type="p1")

            # Added for avoid failing to upsert data (error 503), it likely occurs when just after created index.
            time.sleep(90)
            
            print("Created the new index in Pinecone.")
        except Exception as e:
            print(f"Failed to create index: {e}\n\nTry again by option 2.")
            return False
    else:
        confirmation = input("\nAre you sure to overwrite the exsiting Index? (yes/no): ")
        if confirmation.lower() != "yes":
            print("\nAborted.")
            return False
        # Delete exsisting data in DB
        index.delete(deleteAll='true')

    data = read_jsonl(filename_json)
    if data == None:
        print("\nError: A temporary file does not exist. Please select option 1.")
        return False

    # Upsert Pinecone DB
    print("\nStart upsert...")
    try:
        for i, record in enumerate(data):
            metadata = {key: record[key] for key in record if key != "embeddings"}
            vector = (
                f"id_{i}",
                record["embeddings"],
                metadata
            )
            try:
                index.upsert([vector])
                print(f"id_{i}")
            except Exception as e:
                print(f"Failed to upsert data at index {i}: {e}")
                return False

        print("Completed upsert to Pinecone database.\n\nExit\n")
        return True
    
    except Exception as e:
        print(f"\nFailed to initiate Pinecone index or other error occurred: {e}\n\nPlease select Option 2 to resume it.")
        return False


## Flattens a nested json file. (For future use)
def flatten_json(any_json, delimiter='_'):
    flat_json = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + delimiter)
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + delimiter)
                i += 1
        else:
            flat_json[name[:-1]] = x

    flatten(any_json)
    return flat_json


# Import file -> select a column which should be embeddings -> embeddings -> save it in a temp. file
def extract_inputfile():
    filename = './data/' + input("\nEnter the filename: ")

    if filename.endswith('.csv'):
        try:
            with open(filename, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames

                # Show all available columns
                print("\nAvailable columns:")
                for i, header in enumerate(headers):
                    print(f'{i+1}: {header}')

                # Select a column which should be embeddings
                while True:
                    column_choice = input("Enter the column number you want to create embeddings or type 'menu' to go back: ")
                    if column_choice.lower() == "menu":
                        return False
                    try:
                        column_index = int(column_choice) - 1
                        if column_index >= 0 and column_index < len(headers):
                            selected_column = headers[column_index]
                            print(f"You selected column: {selected_column}")

                            # Confirmation to start embeddings
                            confirmation = input("\nAre you sure to start the embedding process? (yes/no): ")
                            if confirmation.lower() != "yes":
                                    print("\nAborted.")
                                    return False
                            
                            break
                        else:
                            print("\nError: Invalid column number.")
                    except ValueError:
                        print("\nError: Invalid input. Please enter a valid column number.")

                # To save embeddings
                record_embeddings(reader, selected_column, clear_file=True)
                print("Completed.")
            
        except FileNotFoundError:
            print("\nError: File not found. Please enter a valid filename.")
            return False

    else:
        print("Please select a CSV file.")
        return False


while True:
    print("\n\nOptions:")
    print("1: Start from the first step")
    print("2: Resume importing to VectorDB")
    print("3: Quit")
    option = input("\nPlease choose an option: ")

    if option == '1':
        success = extract_inputfile()
        if success:
            vectordb()
    elif option == '2':
        vectordb()
    elif option == '3':
        print("Exit")
        exit(0)
    else:
        print("\nError: Invalid option, please try again.")