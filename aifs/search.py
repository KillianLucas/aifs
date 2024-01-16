"""
Simple, fast, local semantic search. Uses an _.aifs file to store embeddings in top most directory.
"""

# TODO
# Should use system search, like spotlight, to narrow it down. Then rerank with semantic.
# Should use sub indexes in nested dirs if they exist.
# Better chunking that works per sentence, paragraph, word level rather than by character.

import os
import chromadb
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as setup_embed
import json

MAX_CHARS_PER_CHUNK = 500
MAX_CHUNKS = 300 # More than this, and we'll just embed the filename. None to embed any # of chunks.

# Set up the embedding function
os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"  # Otherwise setup_embed displays a warning message
try:
    embed = setup_embed()
except:
    # This does set up a model that we don't strictly need.
    # If it fails, it's not worth breaking everything.
    pass

def chunk_file(path):
    elements = partition(filename=path)
    chunks = chunk_by_title(elements, max_characters=MAX_CHARS_PER_CHUNK)
    return [c.text for c in chunks]

def index_file(path):
    print(f"Indexing {path}...")
    try:
      chunks = chunk_file(path)
      if chunks == []:
        raise Exception("Failed to chunk.")
      if MAX_CHUNKS and len(chunks) > MAX_CHUNKS:
        raise Exception("Too many chunks. Will just embed filename.")
    except Exception as e:
      print(f"Couldn't read `{path}`. Continuing.")
      print(e)
      chunks = [f"There is a file at `{path}`."]

    embeddings = embed(chunks)
    last_modified = os.path.getmtime(path)

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "last_modified": last_modified,
    }

def index_directory(path):
    index = {}
    for root, _, files in os.walk(path):
        for file in files:
            if file != "" and file != "_index.aifs":
                file_path = os.path.join(root, file)
                file_index = index_file(file_path)
                index[file_path] = file_index
    return index

def search(query, path=None, max_results=5):
    if path == None:
        path = os.getcwd()

    path_to_index = os.path.join(path, "_.aifs")
    if not os.path.exists(path_to_index):
        # No index. We're embedding everything.
        print(f"Indexing `{path}` for AI search. This will take time, but only happens once.")
        index = index_directory(path)
        with open(path_to_index, 'w') as f:
            json.dump(index, f)
    else:
        with open(path_to_index, 'r') as f:
            index = json.load(f)
        
        for file_path, file_index in index.items():
            if os.path.getmtime(file_path) != file_index["last_modified"]:
                print(f"Re-indexing {file_path} due to modification.")
                new_file_index = index_file(file_path)
                index[file_path] = new_file_index
        
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="temp")
    id_counter = 0
    for file_path, file_index in index.items():
        ids = [str(id) for id in range(id_counter, id_counter + len(file_index["chunks"]))]
        id_counter += len(file_index["chunks"])
        collection.add(
            ids=ids,
            embeddings=file_index["embeddings"],
            documents=file_index["chunks"],
            metadatas=[{"source": file_path}] * len(file_index["chunks"]),
        )

    results = collection.query(
        query_texts=[query],
        n_results=max_results
    )

    chroma_client.delete_collection("temp")

    return results["documents"][0]