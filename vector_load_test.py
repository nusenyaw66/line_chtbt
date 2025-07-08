import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")

db_name = "hugging_face_FAISS_with_metadata"
# db_name = "hugging_face_chroma_with_metadata"

persistent_db = os.path.join(db_dir, db_name)

def load_vector_store(db_path, embeddings):
    """
    Load a FAISS vector store from a local directory.
    
    Args:
        db_path (str): Path to the saved FAISS index directory.
        embeddings: The embedding model used to create the vector store.
    
    Returns:
        FAISS: Loaded vector store object, or None if loading fails.
    """
    try:
        # Load the vector store
        vector_store = FAISS.load_local(
            folder_path=db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # Required for loading pickled metadata
        )
        print(f"Vector store loaded from {db_path}")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def load_chroma_vector_store(db_path, embeddings):
    """
    Load a Chroma vector store from a local directory.
    
    Args:
        db_path (str): Path to the saved Chroma database directory.
        embeddings: The embedding model used to create the vector store.
    
    Returns:
        Chroma: Loaded vector store object, or None if loading fails.
    """
    try:
        # Load the vector store
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        print(f"Vector store loaded from {db_path}")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def main():
    
    # Initialize embeddings (must match the model used to create the vector store)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Change to "cuda" for GPU
    )
    
    # Load vector store from the persistent directory with Chroma or FAISS
    vector_store = load_vector_store(persistent_db, embeddings)
    # vector_store = load_chroma_vector_store(persistent_db, embeddings)
    
    # Test the vector store (optional)
    if vector_store:
        query = "什麼是美色光？"
        results = vector_store.similarity_search(query, k=3)
        print("\nQuery Results:")
        for doc in results:
            print(f"Index: {doc.metadata['index']}") 
            print(f"Question: {doc.page_content}")
            # print(f"Answer: {doc.metadata['answer']}")
            print(f"Category: {doc.metadata['category']}")
            print(f"Source: {doc.metadata.get('source_file', 'N/A')}\n")

if __name__ == "__main__":
    main()