# Required Packages should be installed on system
"""
Python ver~3.10.12
!pip install langchain==0.0.302
!pip install chromadb==0.4.13
!pip install pdfplumber==0.10.2

!pip install tiktoken==0.5.1
!pip install lxml==4.9.3
!pip install torch==2.0.1
!pip install transformers==4.33.2
!pip install accelerate==0.23.0
!pip install sentence-transformers==2.2.2
!pip install einops==0.6.1
!pip install xformers==0.0.21

!pip install InstructorEmbedding==1.0.1"""

from langchain.document_loaders import PDFPlumberLoader,  TextLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import pipeline
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from chromadb.errors import InvalidDimensionException
import torch
from transformers import AutoTokenizer
import re
import os

app = Flask(__name__)

# Embedding model
EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(EMB_SBERT_MPNET_BASE)

LLM_FLAN_T5_BASE = "google/flan-t5-base"

config = {"persist_directory":None,
          "load_in_8bit":False,
          "embedding" : EMB_SBERT_MPNET_BASE,
          "llm":LLM_FLAN_T5_BASE,
          }

def create_sbert_mpnet():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})


def create_flan_t5_base(load_in_8bit=False):
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=500,
            do_sample = True, #check this
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.5}
        )

embedding = create_sbert_mpnet()
load_in_8bit = config["load_in_8bit"]
llm = create_flan_t5_base(load_in_8bit=load_in_8bit)

# Pre-Load the PDF document and prepare data
pdf_path = "/content/The 2011 Cricket World Cup.pdf" #Set your file path here
loader = PDFPlumberLoader(pdf_path)
documents = loader.load()

try:
    docsearch = Chroma.from_documents(documents=documents, embedding=embedding)
except InvalidDimensionException:
    Chroma().delete_collection()
    docsearch = Chroma.from_documents(documents=documents, embedding=embedding)

# Split documents and create text snippets
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
texts = text_splitter.split_documents(texts)

persist_directory = config["persist_directory"]
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

hf_llm = HuggingFacePipeline(pipeline=llm)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff", retriever=retriever)

@app.route('/answer', methods=['POST'])
def get_answer():
    try:
        data = request.json
        
        question = data.get('question', '')

        qa.combine_documents_chain.verbose = True
        qa.return_source_documents = True
        res = qa({"query": question})

        return jsonify({"answer": res['result']}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()