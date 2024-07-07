"""
Simple, fast, local semantic search. Uses an _.aifs file to store embeddings in top most directory.
"""

import sys

MAX_CHARS_PER_CHUNK = 500
MAX_CHUNKS = None # More than this, and we'll just embed the filename. None to embed any # of chunks.

try:
    from unstructured.chunking.title import chunk_by_title
    from unstructured.partition.auto import partition

    def chunk_file(file_path):
        elements = partition(filename=file_path)
        chunks = chunk_by_title(elements, max_characters=MAX_CHARS_PER_CHUNK)
        return [c.text for c in chunks]

except ImportError:
    # print("Please run 'pip install \"unstructured[all-docs]\"' to improve document-reading performance.")

    def chunk_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            file_text = file.read()
        
        # Chunk the text
        chunks = [file_text[i:i+MAX_CHARS_PER_CHUNK] for i in range(0, len(file_text), MAX_CHARS_PER_CHUNK)]
        return chunks


# TODO
# Should use system search, like spotlight, to narrow it down. Then rerank with semantic.
# Should use sub indexes in nested dirs if they exist.
# Better chunking that works per sentence, paragraph, word level rather than by character.

import ast
import os
from typing import List
import json

try:
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as setup_embed
    # Set up the embedding function
    os.environ[
        "TOKENIZERS_PARALLELISM"
    ] = "false"  # Otherwise setup_embed displays a warning message

    embed = setup_embed()
except:
    pass # shoulda used vlite


# Function to extract function arguments and annotations
def format_function_details(func_def, class_name=None):
    name = func_def.name
    args = [(arg.arg, None if not arg.annotation else ast.unparse(arg.annotation)) for arg in func_def.args.args]
    vararg = (func_def.args.vararg.arg, ast.unparse(func_def.args.vararg.annotation)) if func_def.args.vararg and func_def.args.vararg.annotation else None
    return_annotation = ast.unparse(func_def.returns) if func_def.returns else None
    docstring = ast.get_docstring(func_def)

    # Start with the function name and opening parenthesis
    if class_name:
        formatted_string = f"{class_name}.{name}("
    else:
        formatted_string = f"{name}("
    
    # Add each positional argument with its annotation
    for arg_name, arg_annotation in args:
        formatted_string += f"    {arg_name}: {arg_annotation}, "
    
    # Add varargs if present
    if vararg:
        formatted_string += f"    *{vararg[0]}: {vararg[1]}"
    
    # Close the parenthesis and add return annotation
    formatted_string += f") -> {return_annotation}"
    
    # Add the docstring if present
    if docstring:
        formatted_string += f"  # {docstring}"
    print(formatted_string)
    
    return formatted_string

def log(str):
    verbose = os.environ['LOG_VERBOSE']
    if verbose and verbose == 'True': print(str)

def index_file(file_path, python_docstrings_only=False):
    if python_docstrings_only and file_path.lower().endswith(".py"):
        return minimally_index_python_file(file_path)

    log(f"Indexing {file_path}...")
    try:
      chunks = chunk_file(file_path)
      if chunks == []:
        raise Exception("Failed to chunk.")
      if MAX_CHUNKS and len(chunks) > MAX_CHUNKS:
        raise Exception("Too many chunks. Will just embed filename.")
    except Exception as e:
      log(f"Couldn't read `{file_path}`. Continuing.")
      log(e)
      chunks = [f"There is a file at `{file_path}`."]

    embeddings = embed(chunks)
    last_modified = os.path.getmtime(file_path)

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "last_modified": last_modified,
    }


def minimally_index_python_file(file_path):
    """
    This function indexes a Python file in a minimal way.
    It only embeds the docstrings for semantic search, which then point to only the function name.
    """
    chunks = []
    representations = []
    
    try:
        with open(file_path, "r") as source:
            tree = ast.parse(source.read())
    except Exception as e:
        print(f"Couldn't parse {file_path}. Error:", str(e))
        return None

    def traverse(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                if child.name.startswith("_"):
                    # ignore private functions
                    continue
                class_name = node.name if isinstance(node, ast.ClassDef) else None
                docstring = ast.get_docstring(child)
                formatted_string = format_function_details(child, class_name)
                chunks.append(formatted_string)
                representations.append(docstring if docstring else child.name)
            traverse(child)

    traverse(tree)

    embeddings = embed(representations) if representations else []
    last_modified = os.path.getmtime(file_path)

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "last_modified": last_modified,
    }


def index_files(file_paths, existingIndex=None, indexPath="", python_docstrings_only=False):
    if existingIndex is None:
        existingIndex = {}
    index = existingIndex
    deletedFiles = handle_deleted_files(index)
    modifiedFiles = handle_modified_files(index, python_docstrings_only)
    writeToIndex = len(deletedFiles) > 0 or len(modifiedFiles) > 0
    
    for file_path in file_paths:
        if file_path.endswith("_index.aifs") or file_path.endswith(".DS_Store") or file_path.endswith("_.aifs"):
            continue
        # if there are new files not in index, or modified Files, index them
        if file_path not in index or file_path in modifiedFiles:
            log(f"{file_path} is new file or modified, indexing it")
            writeToIndex = True
            file_index = index_file(file_path, python_docstrings_only)
            if file_index:
                index[file_path] = file_index
        else:
            log(f"{file_path} is in index, skip")
    
    save_index(writeToIndex, index, indexPath)
    
    return index


def handle_deleted_files(index: dict) -> List[str]:
    """updates the index to remove deleted files. returns deleted files"""
    deletedFiles = []
    for file_path, _ in index.items():
        if not os.path.isfile(file_path):
            log(f"Removing {file_path} since it does not exist.")
            deletedFiles.append(file_path)
    # remove deleted files
    for file_path in deletedFiles:
        index.pop(file_path, None)

    return deletedFiles


def handle_modified_files(index: dict, python_docstrings_only: bool) -> List[str]:
    modifiedFiles = []
    for file_path, file_index in index.items():
        if file_path == None or file_index == None:
            continue
        if os.path.getmtime(file_path) != file_index["last_modified"]:
            log(f"Re-indexing {file_path} due to modification.")
            new_file_index = index_file(file_path, python_docstrings_only)
            index[file_path] = new_file_index
            modifiedFiles.append(file_path)
    return modifiedFiles


def save_index(writeToIndex, index, indexPath):
    if writeToIndex:
        log(f"Index has changed, saving again to {indexPath}")
        with open(indexPath, "w") as f:
            json.dump(index, f)


def index_directory(path, existingIndex=None, indexPath="", python_docstrings_only=False):
    if existingIndex is None:
        existingIndex = {}
    index = existingIndex
    deletedFiles = handle_deleted_files(index)
    modifiedFiles = handle_modified_files(index, python_docstrings_only)
    writeToIndex = len(deletedFiles) > 0 or len(modifiedFiles) > 0
    
    for root, _, files in os.walk(path):
        for file in files:
            if file != "" and file != "_index.aifs" and file != ".DS_Store" and file != "_.aifs":
                file_path = os.path.join(root, file)
                # if there are new files not in index, or modified Files, index them
                if file_path not in index or file_path in modifiedFiles:
                    log(f"{file_path} is new file or modified, indexing it")
                    writeToIndex = True
                    file_index = index_file(file_path, python_docstrings_only)
                    index[file_path] = file_index
                else:
                   log(f"{file_path} is in index, skip")
    
    save_index(writeToIndex, index, indexPath)
    
    return index


def search(query, path=None, file_paths=None, max_results=5, verbose=False, python_docstrings_only=False):
    """
    Performs a semantic search of the `query` in `path` and its subdirectories.

    Parameters:
    query (str): The search query.
    path (str, optional): The path to the directory to search. Defaults to the current working directory.
    file_paths (list, optional): A list of file paths to search. Defaults to None. Used only if path isn't provided.
    max_results (int, optional): The maximum number of search results to return. Defaults to 5.

    Returns:
    list: A list of search results.
    """
    os.environ['LOG_VERBOSE'] = str(verbose)

    if path is None:
        common_prefix = os.path.commonprefix(file_paths)
        if not common_prefix.endswith("/"):
            common_prefix = os.path.dirname(common_prefix)
        path_to_index = os.path.join(common_prefix, "_.aifs")
    else:
        path_to_index = os.path.join(path, "_.aifs")

    index = {}
    if not os.path.exists(path_to_index):
        # No index. We're embedding everything.
        log(f"Indexing for AI search. This will take time, but only happens once.")
    else:
        log(f"Using existing index at `{path_to_index}`")
        with open(path_to_index, 'r') as f:
            index = json.load(f)
    
    if path or file_paths is None:
        if path is None:
            path = os.getcwd()
        index = index_directory(path, existingIndex=index, indexPath=path_to_index, python_docstrings_only=python_docstrings_only)
    else:
        index = index_files(file_paths, existingIndex=index, indexPath=path_to_index, python_docstrings_only=python_docstrings_only)

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="temp")
    id_counter = 0
    for file_path, file_index in index.items():
        if "__pycache__" in file_path:
            continue
        if file_index and "chunks" in file_index:
            ids = [str(id) for id in range(id_counter, id_counter + len(file_index["chunks"]))]
            id_counter += len(file_index["chunks"])
        else:
            ids = []
        if ids:
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
    log(results)

    chroma_client.delete_collection("temp")

    return results["documents"][0]
