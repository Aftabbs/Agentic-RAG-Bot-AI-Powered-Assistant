# setup_rag_bot.py
"""
Setup script for RAG-Enhanced Real Estate Bot
This script will help you set up everything needed for the bot
""" 
 
import os
import subprocess
import sys

def install_requirements():
    """Install required packages""" 
    print("Installing required packages...")
    
    packages = [
        "google-generativeai",
        "langchain",
        "chromadb",
        "pypdf",
        "sentence-transformers",
        "tiktoken",
        "python-dotenv",
        "requests",
        "unstructured[docx]"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nAll packages installed successfully!")

def create_env_file():
    """Create .env file template"""
    if not os.path.exists(".env"):
        print("\nCreating .env file...")
        env_content = """# API Keys for Real Estate Bot
GEMINI_API_KEY=your_gemini_api_key_here
SERPER_API_KEY=your_serper_api_key_here
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print(".env file created. Please add your API keys.")
    else:
        print(".env file already exists.")

def create_utils_file():
    """Create utils.py with prompt templates"""
    print("\nCreating utils.py...")
    
    utils_content = '''"""
Utility functions for Real Estate Bot
"""

def get_real_estate_system_prompt():
    """Returns the system prompt for the real estate agent"""
    return """You are Agent Mira, an expert real estate AI assistant specializing in South Florida real estate.
    
Your expertise includes:
- Detailed knowledge of Miami-Dade and Broward County neighborhoods
- Current market trends and investment strategies
- Home buying and selling processes
- Property valuation and analysis
- Mortgage and financing options
- Legal and regulatory requirements

Personality traits:
- Professional yet friendly
- Patient and thorough in explanations
- Proactive in providing valuable insights
- Honest about market conditions
- Focused on client's best interests

Always provide specific, actionable advice and cite actual neighborhoods, price ranges, and market data when relevant."""

def get_validator_system_prompt():
    """Returns the system prompt for the validator agent"""
    return """You are a validator for a real estate assistant. Your job is to ensure queries are appropriate.

For each query, respond with either:
- "VALID" if the query is related to real estate, property, neighborhoods, market analysis, home buying/selling, mortgages, or general conversation
- "INVALID: [brief redirect message]" if the query is completely unrelated to real estate

Be lenient - general greetings, small talk, and tangentially related topics should be marked as VALID.
Only mark clearly off-topic requests (like cooking recipes, medical advice, etc.) as INVALID."""
'''
    
    with open("utils.py", "w") as f:
        f.write(utils_content)
    print("utils.py created successfully!")

def create_update_script():
    """Create a script to update the knowledge base"""
    print("\nCreating update_knowledge.py...")
    
    update_content = '''"""
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
'''
    
    with open("update_knowledge.py", "w") as f:
        f.write(update_content)
    print("update_knowledge.py created successfully!")

def main():
    """Main setup function"""
    print("="*60)
    print("RAG-Enhanced Real Estate Bot Setup")
    print("="*60)
    
    # Step 1: Install requirements
    response = input("\nDo you want to install required packages? (y/n): ")
    if response.lower() == 'y':
        install_requirements()
    
    # Step 2: Create .env file
    create_env_file()
    
    # Step 3: Create utils.py
    create_utils_file()
    
    # Step 4: Create update script
    create_update_script()
    
    # Step 5: Create directories
    print("\nCreating directory structure...")
    dirs = [
        "./real_estate_knowledge",
        "./new_documents",
        "./processed_documents"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created {dir_path}")
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Add your API keys to the .env file")
    print("2. Add knowledge documents to ./real_estate_knowledge/")
    print("3. Run: python rag_enhanced_bot.py")
    print("\nTo update knowledge base later:")
    print("- Add new documents to ./new_documents/")
    print("- Run: python update_knowledge.py")
    print("\nHappy chatting with Agent Mira!")

if __name__ == "__main__":

    main()


