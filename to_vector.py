#!/usr/bin/env python3

# This script is for auto importing data from a CSV file with multiple columns into a vector database.

import os
import pandas as pd
import openai
import pinecone
import csv
import json
import copy
import time
from dotenv import load_dotenv


# Load the .env.local file
load_dotenv('.env.local')
openai.api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")

# Define the embeddings model
EMBEDDING_MODEL = "text-embedding-ada-002"

# Define temporary file name and location
default_temp_file = './data/temp/temp.jsonl'

# Create the directory if it does not exist
dir_name = os.path.dirname(default_temp_file)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# For importing jsonl data
def read_jsonl(filename_jsonl):
    try:
        with open(filename_jsonl, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print("\nTemporaly file not found.")
        return False

# For saving embeddings to a specified file. 
# clear_file: {True: Clears existing data and overwrites it, False: Appends new data after the existing data.}
def record_embeddings(data_set, embedding_column_name, identifier_column, clear_file, output_file_name = default_temp_file):
    if clear_file:
        # Open the file in write mode to create it or clear it if it already exists
        with open(output_file_name, 'w', encoding='utf-8') as json_file:
            pass

    with open(output_file_name, 'a', encoding='utf-8') as json_file:
        print("\nStart embeddigns process...")
        counter = 1
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
            
            print(f"id_{row[identifier_column]}")
            row_dict = row.copy()
            row_dict["embeddings_target"] = embedding_column_name
            row_dict["embeddings"] = embedding
            
            json.dump(row_dict, json_file, ensure_ascii=False)
            json_file.write("\n")

            counter += 1

# For saving Vector and meta data in online vectorDB(s)
def vectordb(filename_json = default_temp_file):
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
            print(f"Failed to create index: {e}\n\nTry again with option 3.")
            return False
    else:
        confirmation = input("\nAre you sure to overwrite the exsiting Index? (yes/no): ")
        if confirmation.lower() != "yes":
            print("\nAborted.")
            return False
        # Delete exsisting data in DB
        index.delete(deleteAll='true')

    # Read data from temp.jsonl file.
    data = read_jsonl(filename_json)
    if data == None:
        print("\nError: A temporary file does not exist. Please select option 1.")
        return False

    # Upsert Pinecone DB
    print("\nStart upsert...")
    try:
        for i, record in enumerate(data):
            metadata = {key: record[key] for key in record if key not in ["embeddings", "embeddings_target"]}
            id_num = i+1
            vector = (
                f"id_{id_num}",
                record["embeddings"],
                metadata
            )
            try:
                index.upsert([vector])
                print(f"id_{id_num}")
            except Exception as e:
                print(f"Failed to upsert data at index {id_num}: {e}")
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


# Check difference between two files
def find_diff(dataset_new, dataset_old, identifier_column, output_file_name = default_temp_file):
    # Find common and different ReportIDs
    dataset_new_ids = {row[identifier_column] for row in dataset_new}
    dataset_old_ids = {row[identifier_column] for row in dataset_old}
    common_ids = dataset_new_ids.intersection(dataset_old_ids)
    different_ids = dataset_new_ids.symmetric_difference(dataset_old_ids)
    # Initiate a list to hold missing data
    missing_data = []

    # Check if contents of common_ids are the same in both datasets
    len_common_ids = len(common_ids)
    if len_common_ids > 1:
        # Initiate a list to hold unmatched IDs
        unmatched_ids = []
        
        for id_ in common_ids:
            # Subset the data based on the id and sort by sort_by_column
            new_subset = sorted([row for row in dataset_new if row[identifier_column] == id_], key=lambda x: x[identifier_column])
            
            # Make a copy of old subset to preserve the original data with 'embeddings'
            old_subset = [row for row in dataset_old if row[identifier_column] == id_]
            old_subset_copy = copy.deepcopy(old_subset)
            for item in old_subset_copy:
                item.pop('embeddings_target', None)
                item.pop('embeddings', None)
            old_subset_copy = sorted(old_subset_copy, key=lambda x: x[identifier_column])

            # Check if the subsets match, and then add the data from the new_subset to the missing_data list
            if new_subset != old_subset_copy:
                unmatched_ids.append(id_)
                missing_data.extend(new_subset)

        # Print all unmatched ids together
        print(f"\nUpdated data ID: {', '.join(map(str, sorted(unmatched_ids)))}")

        # Remove the unmatched ids from the old_dataset and save it to temp,jsol file.
        remove_ids = unmatched_ids + list(different_ids)  # IDs to be removed from old dataset
        dataset_old = [row for row in dataset_old if row[identifier_column] not in remove_ids]
        with open(output_file_name, 'w', encoding='utf-8') as f:
            for entry in dataset_old:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

    len_diff_ids = len(different_ids)
    if len_diff_ids > 1:
        sorted_ids = sorted(list(different_ids))
        print('Nonexistent Data ID:', ', '.join(map(str, sorted_ids)))

    # Find rows in CSV (dataset_new) that are not in JSONL (dataset_old)
    missing_data.extend([row for row in dataset_new if row[identifier_column] in different_ids])
    return missing_data

# Import file -> select a column which should be embeddings -> embeddings -> save it in a temp. file
def extract_inputfile(append):
    filename = './data/' + input("\nEnter the filename: ")

    if filename.endswith('.csv'):
        try:
            with open(filename, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                identifier_column = headers[0]
                clear_file=True
                embeddings_target = None

                # To pickup diff data if append is true.
                if append:
                    # Check if default temp. file is exsits
                    if not os.path.exists(default_temp_file):
                        print("\nTemporaly file not found. Please try again with option 1.")
                        return False
                    csv_data = list(reader)
                    jsonl_data = read_jsonl(default_temp_file)
                    reader = find_diff(csv_data, jsonl_data, identifier_column)
                    # Retrieve 'embeddings_target' from the first data entry
                    embeddings_target = jsonl_data[0].get('embeddings_target', None)
                    # Switching to append mode for def "record_embedding"
                    clear_file=False

                # Show all available columns and select embeddings_target if it is None.
                if embeddings_target is None:
                    print("\nAvailable columns:")
                    for i, header in enumerate(headers):
                        print(f'{i+1}: {header}')

                    # Select a column which should be embeddings
                    while True:
                        column_choice = input("Enter the column number you want to create embeddings OR type 'menu' to go back: ")
                        if column_choice.lower() == "menu":
                            return False
                        try:
                            column_index = int(column_choice) - 1
                            if column_index >= 0 and column_index < len(headers):
                                embeddings_target = headers[column_index]
                                break
                            else:
                                print("\nError: Invalid column number.")
                        except ValueError:
                            print("\nError: Invalid input. Please enter a valid column number.")

                # Confirmation to start embeddings
                print(f"\nColumn to be embedded: {embeddings_target}")
                confirmation = input("\nAre you sure to start the embedding process? (yes/no): ")
                if confirmation.lower() != "yes":
                    print("\nAborted.")
                    return False
                record_embeddings(reader, embeddings_target, identifier_column, clear_file)
                print("Completed.")
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
    print("2: Append / Replace Data")
    print("3: Resume importing to VectorDB")
    print("4: Quit")
    option = input("\nPlease choose an option: ")

    if option == '1':
        if extract_inputfile(append=False):
            vectordb()
    elif option == '2':
        if extract_inputfile(append=True):
            vectordb()
    elif option == '3':
        vectordb()
    elif option == '4':
        print("Exit")
        exit(0)
    else:
        print("\nError: Invalid option, please try again.")