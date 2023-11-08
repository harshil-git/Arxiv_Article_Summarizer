import os
from constants import openai_key
import streamlit as st
import arxiv
import ast
import concurrent
from csv import writer
import json
import openai
import pandas as pd
from PyPDF2 import PdfReader
import requests
import tiktoken
import pickle
import time
from langchain import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate


os.environ["OPENAI_API_KEY"]=openai_key
openai.api_key = openai_key
directory = './data/papers'
# Check if the directory already exists
if not os.path.exists(directory):
    # If the directory doesn't exist, create it and any necessary intermediate directories
    os.makedirs(directory)
    print(f"Directory '{directory}' created successfully.")
else:
    # If the directory already exists, print a message indicating it
    print(f"Directory '{directory}' already exists.")


# Set a directory to store downloaded papers
data_dir = os.path.join(os.curdir, "data", "papers")
dir_csv_filepath = "./data/arxiv_data.csv"

#openai.api_key = openai_key
def article_search(query,max_results=5,library=dir_csv_filepath):
   
   search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
   result_list = []
   
   for result in arxiv.Client().results(search):
        result_dict = {}
        result_dict.update({"title": result.title})
        result_dict.update({"summary": result.summary})

        result_dict.update({"article_url": [x.href for x in result.links][0]})
        result_dict.update({"pdf_url": [x.href for x in result.links][1]})
        result_list.append(result_dict)
   df = pd.DataFrame(result_list)
   df.to_csv(library)    
   return result_list
  
  

st.title('Arxiv Article Summarizer 📝')
faiss_index_pickle_path = "faiss_index_path.pkl"

query = st.text_input("Question: ")


if query:
    article_search(query)
    #load data from .csv file
    loader = CSVLoader(dir_csv_filepath,source_column="article_url")
    st.text("Data loading started....✅✅")
    data = loader.load()

    #split data
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n','\n','.',','], chunk_size=400)
    st.text("Text splittter started....✅✅")
    docs = text_splitter.split_documents(data)

    #create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vector_faiss_openai = FAISS.from_documents(docs,embeddings)
    st.text("Embedding vectors and faiss index assignment started ...✅✅")
    time.sleep(2)

    #saving the FAISS index to the pickle file
    with open(faiss_index_pickle_path, "wb") as f:
        pickle.dump(vector_faiss_openai,f)
    if os.path.exists(faiss_index_pickle_path):
        with open(faiss_index_pickle_path, "rb") as f:
            vectorstore = pickle.load(f)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613",
             messages= [{"role" : "user","content":f"""Use the following key points to summarize the user query:{query}. 
summary should highlight core argument, conclusions and evidence. 
summary should be in bullet points followed by headings core argument, evidence and conclusion.
                         Key points:\n{vectorstore}\nSummary:\n"""}],temperature=0.9,max_tokens=400)
    
    st.header("Answer")
    st.write(response["choices"][0]["message"]["content"])
    
            






