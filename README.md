
# Arxiv Article Summarizer

## Overview
This is a Arxiv Article Summarizer app, summarizes the Articles of Arxiv library based on the desired topic. It pulls the articles from Arxiv library, splits content into chunks with the help of Langchain Document loaders and transformers and saves as FAISS index for information retrieval. Later, based on the topic given and created FAISS index, GPT 3.5 summarizes content in the form of core argument, evidence and conclusions and present to the user. 



## Features

- uses Langchain's CSVLoader to load article content.
- transforms large unstructured data into chunks using Langchain's RecursiveCharacterTextSplitter
- uses OpenAI Embeddings to create embedding vectors and leverages FAISS library for similarity search and retrieves relevant information.
- Interact with OpenAI ChatGPT and presents answer.


## Run Locally

Clone the project

```bash
  gh repo clone harshil-git/Arxiv_Article_Summarizer
```

Go to the project directory

```bash
  cd Arxiv_Article_Summarizer
```

Install dependencies

```bash
  pip install -r requirements.txt
```
 
Set your OpenAI API key in constants.py file.

Start the server

```bash
  streamlit run main.py 
```


## Usage/Examples

1)  Start the server

```bash
  streamlit run main.py 
```
2) Web app will open in your browser.
3) Search the topic or title.
4) It will load the data based on the searched question and splits the retrieved text, generates embedding vectors and indexes them using FAISS. It will present the answer.

## Project Structure

##### main.py : Streamlit application script.
##### requirements.txt : list of python packages to install     for this project.
##### faiss_index_path.pkl : for storing FAISS index.
##### constants.py : storing OpenAI API key


