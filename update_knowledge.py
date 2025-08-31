"""
Script to update the RAG knowledge base with new documents
"""

import os
import sys
from document_processor import DocumentProcessor

def update_knowledge_base(new_docs_path="./new_documents"):
    """
    Add new documents to the existing knowledge base
    
    Args:
        new_docs_path: Path to directory containing new documents
    """
    if not os.path.exists(new_docs_path):
        print(f"Creating {new_docs_path} directory...")
        os.makedirs(new_docs_path)
        print(f"Please add your new documents to {new_docs_path} and run this script again.")
        return
    
    # Check if there are any documents
    files = os.listdir(new_docs_path)
    docs = [f for f in files if f.endswith(('.pdf', '.txt', '.docx'))]
    
    if not docs:
        print(f"No documents found in {new_docs_path}")
        print("Please add PDF, TXT, or DOCX files and try again.")
        return
    
    print(f"Found {len(docs)} documents to process")
    
    # Process documents
    processor = DocumentProcessor()
    documents = processor.load_documents(new_docs_path)
    
    if documents:
        # Add to existing vector store
        from langchain.vectorstores import Chroma
        
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=processor.embeddings
        )
        
        # Split and add new documents
        texts = processor.text_splitter.split_documents(documents)
        vectorstore.add_documents(texts)
        
        print(f"Successfully added {len(texts)} chunks to knowledge base")
        
        # Move processed files to archive
        archive_dir = "./processed_documents"
        os.makedirs(archive_dir, exist_ok=True)
        
        for doc in docs:
            os.rename(
                os.path.join(new_docs_path, doc),
                os.path.join(archive_dir, doc)
            )
        
        print(f"Moved processed documents to {archive_dir}")
    else:
        print("No documents were loaded. Please check the files.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        update_knowledge_base(sys.argv[1])
    else:
        update_knowledge_base()
