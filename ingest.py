from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_database():
    # Load PDFs
    print("Loading documents from directory...")
    loader = PyPDFDirectoryLoader("data/")
    pages = loader.load()
    print(f"Loaded {len(pages)} total pages.")

    # Chunking
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(pages)
    print(f"Created {len(documents)} chunks.")

    # Embeddings + Vector DB
    print("Building and saving Vector Database...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # save the database to a folder called "chroma_db"
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db" 
    )
    
    print("Database saved to ./chroma_db")

if __name__ == "__main__":
    build_vector_database()