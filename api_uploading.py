from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.DEBUG)



# Initialize FastAPI
app = FastAPI()

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

def save_uploaded_file(uploaded_file: UploadFile, folder_path: str):
    file_path = os.path.join(folder_path, uploaded_file.filename)
    if os.path.exists(file_path):
        logging.info(f"File already exists: {file_path}")
        return False  # File already exists
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.file.read())
    logging.info(f"Saved file: {file_path}")
    return True

def create_or_update_vector_db(folder_path: str, category_folder: str):
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

@app.post("/upload/{file_name}")
async def upload_files(category: str = Form(...), files: List[UploadFile] = File(...)):
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid category")

    folder_name = CATEGORIES[category]
    folder_path = os.path.join(DATA_PATH, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    files_uploaded = 0
    for uploaded_file in files:
        if save_uploaded_file(uploaded_file, folder_path):
            files_uploaded += 1

    if files_uploaded == 0:
        return {"message": "All uploaded files already exist in the folder."}
    else:
        # Create or update FAISS index
        result = create_or_update_vector_db(folder_path, folder_name)
        return result

@app.get("/")
def read_root():
    return {"message": "Welcome to the Data Reader API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
# from fastapi import FastAPI, HTTPException, Query
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import ConversationalRetrievalChain
# from langchain_ollama import ChatOllama, OllamaEmbeddings
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# # Define paths and configurations
# BASE_PATH = "D:/insights trail/datasheet/"
# DB_FAISS_PATH = os.path.join(BASE_PATH, "faiss_index")

# # Initialize FastAPI
# app = FastAPI()

# # Initialize LLM
# llm = ChatOllama(
#     model="llama3.1,
#     temperature=0,
# )

# # Define prompt template
# prompt_template = """Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question.
# <context>
# {context}
# </context>
# Questions: {input}
# """
# prompt = ChatPromptTemplate.from_template(template=prompt_template)

# # Embeddings configuration
# embeddings = OllamaEmbeddings(model="llama3.1")

# def find_file(file_name):
#     for drive in ["C:\\", "D:\\"]:  # Add more drives as needed
#         for root, dirs, files in os.walk(drive):
#             if file_name in files:
#                 return os.path.join(root, file_name)
#     raise HTTPException(status_code=404, detail="File not found")

# def create_vector_store(file_path, category):
#     db_folder_path = os.path.join(DB_FAISS_PATH, category)
#     os.makedirs(db_folder_path, exist_ok=True)
#     db_path = os.path.join(db_folder_path, 'db_faiss')

#     loader = PyPDFLoader(file_path)
#     pages = loader.load_and_split()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     all_splits = text_splitter.split_documents(pages)

#     if os.path.exists(db_path):
#         vectorstore = FAISS.load_local(db_path, embeddings)
#         vectorstore.add_documents(all_splits)
#     else:
#         vectorstore = FAISS.from_documents(all_splits, embeddings)

#     vectorstore.save_local(db_path)
#     return db_path

# def load_vector_store(category):
#     db_path = os.path.join(DB_FAISS_PATH, category, 'db_faiss')
#     if not os.path.exists(db_path):
#         raise HTTPException(status_code=404, detail="Vector store does not exist for the specified category.")
#     return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

# def get_response(query, retriever):
#     chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=True)
#     chat_history = []
#     result = chain({"question": query, "chat_history": chat_history})
#     return result['answer']

# @app.post("/upload/")
# async def upload_pdf(category: str, file_name: str):
#     try:
#         file_path = find_file(file_name)
#         db_path = create_vector_store(file_path, category)
#         return {"message": "Vector store created and saved.", "index_path": db_path}
#     except HTTPException as e:
#         raise e

# @app.get("/query/")
# async def answer_query(query: str, category: str):
#     try:
#         vectorstore = load_vector_store(category)
#         retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
#         answer = get_response(query, retriever)
#         return {"answer": answer}
#     except HTTPException as e:
#         raise e

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Data Reader API"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
