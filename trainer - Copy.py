import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai
from dotenv import load_dotenv
load_dotenv()
## load the GROQ And OpenAI API KEY 
#os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
#groq_api_key=os.getenv('GROQ_API_KEY')
groq_api_key='000000000000000000000000000000000000000'
#embeddings=OpenAIEmbeddings()

# st.title("Chatgroq With Llama3 Demo")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
#from ctransformers import AutoModelForCausalLM
## Read the ppdfs from the folder
from ctransformers import AutoModelForCausalLM
file="result.pdf"

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file)
pages = loader.load_and_split()
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(pages)
# storing embeddings in the vector store
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
from langchain.embeddings import HuggingFaceBgeEmbeddings

## Embedding Using Huggingface
huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

)

from langchain_community.vectorstores import FAISS
vectorstore=FAISS.from_documents(all_splits,huggingface_embeddings)
from langchain.chains import ConversationalRetrievalChain

chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
#save locally 
#vectorstore.save_local("faiss_index")

#new_db = FAISS.load_local("faiss_index", huggingface_embeddings)
#new_db = FAISS.load_local("faiss_index", huggingface_embeddings, allow_dangerous_deserialization=True)


#docs = new_db.similarity_search(query)
## Query using Similarity Search
# query="what is Tradition of Historiography ?"
# relevant_docments=vectorstore.similarity_search(query)

# print(relevant_docments[0].page_content)
chat_history = []

query = "What is the difference between mass and weight of an object. Will the mass and weight of an object on the earth be sameas their values on Mars? Why?? "
result = chain({"question": query, "chat_history": chat_history})

print(result['answer'])