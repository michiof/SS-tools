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

# Define the model
model = "text-embedding-ada-002"


# For creating embeddings
def create_embeddings(row, model, column_index):
    try:
        embedding = openai.Embedding.create(input=row[column_index], model=model)["data"][0]["embedding"]
        return embedding
    except Exception as e:
        print(f"Error when creating embeddings in {row}: {e}")
        return None

# For saving Vector and meta data in online vectorDB(s)
def vectordb():
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

    # Import from temp.json data
    try:
        with open('./data/temp2.jsonl', 'r', encoding='utf-8-sig') as f:
            data = [json.loads(line) for line in f.readlines()]
    except FileNotFoundError:
        print(f"A temporary file does not exist. Please select option 1.")
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


## Flattens a nested json file.
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


# Import file -> select a column which should be embeddings -> embeddings -> save it in a temp. file -> vectorDB
def extract_inputfile():
    filename = './data/' + input("\nEnter the filename: ")

    if filename.endswith('.csv'):
        try:
            with open(filename, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                headers = next(reader)

                # Show all available columns
                print("\nAvailable columns:")
                for i, header in enumerate(headers):
                    print(f'{i+1}: {header}')
                print("\n== Please make sure that all column names are written in English. ==\n")
                
                # Select a column which should be embeddings
                column_choice = input("Enter the column number you want to create embeddings: ")
                try:
                    column_index = int(column_choice) - 1
                    if column_index >= 0 and column_index < len(headers):
                        selected_column = headers[column_index]
                        print(f"You selected column: {selected_column}")

                        confirmation = input("\nAre you sure to start the embedding process? (yes/no): ")
                        if confirmation.lower() != "yes":
                            print("\nAborted.")
                            return False
                        
                        # Create/Clear a temporaly json file, and record embedddings with metadata
                        temp_file_name = './data/temp2.jsonl'
                        print(temp_file_name)
                        # Open the file in write mode to create it or clear it if it already exists
                        with open(temp_file_name, 'w', encoding='utf-8-sig') as json_file:
                            pass

                        with open(temp_file_name, 'a', encoding='utf-8-sig') as json_file:
                            print("\nStart embeddigns process for")
                            counter = 1
                            for row in reader:
                                
                                # for avoiding API rate limit, wait 60sec every 5000
                                if counter % 5000 == 0:
                                    print("Waiting...")
                                    time.sleep(60)
                                
                                # If None (an embeddings error) is returned, back to menu.
                                embedding = create_embeddings(row, model, column_index)
                                print(row[0])
                                if embedding is None:
                                    print("Embedding creation failed. Please try agin.")
                                    return False
                                
                                row_dict = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                                row_dict["embeddings"] = embedding
                                
                                json.dump(row_dict, json_file, ensure_ascii=False)
                                json_file.write("\n")

                                counter += 1
                    else:
                        print("\nError: Invalid column number.")
                        return False

                except ValueError:
                    print("\nError: Invalid input. Please enter a valid column number.")
                    return False
                
                print("Completed.")
            
            # To save vectorDB
            vectordb()

            return True

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
            break
    elif option == '2':
        success = vectordb()
        if success:
            break
    elif option == '3':
        print("Exit")
        exit(0)
    else:
        print("\nError: Invalid option, please try again.")