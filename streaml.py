from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS

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
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

groq_api_key='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'





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







from langchain.embeddings import HuggingFaceBgeEmbeddings
## Embedding Using Huggingface
huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

)


#prompt = PromptTemplate(template=prompt, input_variables=['context', 'question'])

new_db = FAISS.load_local("faiss_index", huggingface_embeddings, allow_dangerous_deserialization=True)

chain_type_kwargs = {"prompt": prompt}
#docs = new_db.similarity_search(query)
chain = ConversationalRetrievalChain.from_llm(llm, new_db.as_retriever(), return_source_documents=True)
#retriever = new_db.as_retriever(search_kwargs={"k":1})
# chat_history = []

# query = []
# query = "What is the difference between mass and weight of an object? "

# result = chain({"question": query, "chat_history": chat_history})

# print(result['answer'])
#chain= RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
def main():
    st.title("Conversational Q&A Chatbot")

    # Get user input
    query = st.text_input("Enter your query:")
    
    if st.button("Get Answer"):
        if query:
            # Generate response from the chain
            result = chain({"question": query, "chat_history": []})
            # Display the chatbot's response
            st.write("Bot's Response:")
            st.write(result['answer'])
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()