import streamlit as st
import os
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define paths
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore_pro/'

# Ensure the necessary directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(DB_FAISS_PATH, exist_ok=True)

# Define categories
CATEGORIES = {
    'Civil Law': 'civil_law',
    'Commercial Law': 'commercial_law',
    'Corporate Law': 'corporate_law',
    'Criminal Law': 'criminal_law',
    'Family Law': 'family_law'
}

def save_uploaded_file(uploaded_file, folder_path):
    file_path = os.path.join(folder_path, uploaded_file.name)
    if os.path.exists(file_path):
        logging.info(f"File already exists: {file_path}")
        return False  # File already exists
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    logging.info(f"Saved file: {file_path}")
    return True

def create_or_update_vector_db(folder_path, category_folder):
    logging.info(f"Processing folder: {folder_path}")
    db_folder_path = os.path.join(DB_FAISS_PATH, category_folder)
    db_path = os.path.join(db_folder_path, 'db_faiss')

    os.makedirs(db_folder_path, exist_ok=True)

    loader = DirectoryLoader(folder_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        logging.warning(f"No PDF documents found in {folder_path}.")
        return {"message": f"No PDF documents found in {folder_path}."}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    try:
        if os.path.exists(db_path):
            db = FAISS.load_local(db_path, embeddings)
            db.add_documents(texts)
        else:
            db = FAISS.from_documents(texts, embeddings)

        db.save_local(db_path)
        logging.info(f"Vector database updated and saved at {db_path}")
        return {"message": f"Vector database updated and saved at {db_path}"}
    except Exception as e:
        logging.error(f"Error creating/updating vector database: {e}")
        return {"error": str(e)}

# Streamlit UI
st.title('PDF Processing and FAISS Index Update')

# Radio button for category selection
category = st.radio("Select a category for the PDFs", options=list(CATEGORIES.keys()))

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)

# Folder name input (based on selected category)
folder_name = CATEGORIES.get(category)

if st.button('Upload and Process'):
    if not folder_name:
        st.error("Please select a category.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF file.")
    else:
        # Create folder path
        folder_path = os.path.join(DATA_PATH, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        files_uploaded = 0
        for uploaded_file in uploaded_files:
            if save_uploaded_file(uploaded_file, folder_path):
                files_uploaded += 1
        
        if files_uploaded == 0:
            st.warning("All uploaded files already exist in the folder.")
        else:
            # Create or update FAISS index
            result = create_or_update_vector_db(folder_path, folder_name)
            st.write(result)
