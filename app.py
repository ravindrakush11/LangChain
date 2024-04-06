from langchain.document_loaders import PDFPlumberLoader,  TextLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import pipeline
from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import torch
from transformers import AutoTokenizer
import re
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Embedding model
# EMB_SBERT_MPNET_BASE = 
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

LLM_FLAN_T5_BASE = "google/flan-t5-base"

def create_flan_t5_base(load_in_8bit=False):
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=500,
            #do_sample = True, #check this
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.5}
        )


embedding = model,
persist_directory=None,
# load_in_8bit = ["load_in_8bit"]
llm = create_flan_t5_base()


# Pre-Load the PDF document and prepare data
# pdf_path = "C:\\Users\\Pradeep\\Downloads\\SRS 03.pdf" #Set your file path here
# loader = PDFPlumberLoader(pdf_path)
# documents = loader.load()


# # Split documents and create text snippets
# """Load the document, split it into chunks, embed each chunk and load it into the vector store."""
# """"Split documents into small chunks. This is so we can find the most relevant chunks for a query and pass only those into the LLM."""
# text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)
# text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
# texts = text_splitter.split_documents(texts)

# persist_directory = config["persist_directory"]
# vector_db = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

vector_db = Chroma.from_documents()

hf_llm = HuggingFacePipeline()
retriever = vector_db.as_retriever()
qa = RetrievalQA.from_chain_type()



# hf_llm = HuggingFacePipeline(pipeline=llm)
# retriever = vector_db.as_retriever(search_kwargs={"k": 4})
# qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff", retriever=retriever)



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
    
    
coll = db.get()  # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])

# ids_to_del = []

# for idx in range(len(coll['ids'])):

#     id = coll['ids'][idx]
#     metadata = coll['metadatas'][idx]

#     if metadata['source'] == "source_doc_to_remove":
#         ids_to_del.append(id)

# db._collection.delete(ids_to_del)

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}),400

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}),400

#     if file:
#         # Ensure the file has a PDF extension
#         if file.filename.endswith('.pdf'):
#             # Save the uploaded file to the UPLOAD_FOLDER
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

#             return jsonify({'message': 'File uploaded successfully'}),200
#         else:
#             return jsonify({'error': 'Invalid file format. Please upload a PDF'}),400

@app.route('/uploadAndProcess', methods=['POST'])
def upload_file_and_process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}),400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}),400

    if file:
        # Ensure the file has a PDF extension
        if file.filename.endswith('.pdf'):
            # Save the uploaded file to the UPLOAD_FOLDER
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            # Pre-Load the PDF document and prepare data
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename) 
            #"C:\\Users\\Pradeep\\Downloads\\SRS 03.pdf" #Set your file path here
            loader = PDFPlumberLoader(pdf_path)
            documents = loader.load()
            # try:
            #     docsearch = Chroma.from_documents(documents=documents, embedding=embedding)
            # except InvalidDimensionException:
            #     Chroma().delete_collection()
            #     docsearch = Chroma.from_documents(documents=documents, embedding=embedding)

            # Split documents and create text snippets
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)
           

           
            vector_db = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

            hf_llm = HuggingFacePipeline(pipeline=llm)
            retriever = vector_db.as_retriever(search_kwargs={"k": 4})
            qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff", retriever=retriever)
            return jsonify({'message': 'File uploaded successfully'}),200
        else:
            return jsonify({'error': 'Invalid file format. Please upload a PDF'}),400

if __name__ == '__main__':
    app.run(host='192.168.11.138', port=3200)
    #(host='0.0.0.0', port=8080)