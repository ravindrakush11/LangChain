"""
JSON request structure at /upload-
{
    "filepath": [
        "/content/NLTK.pdf",
        "/content/The 2011 Cricket World Cup.pdf",
        "/content/Vanshita_thesis_soft_copy.pdf",
        "/content/PlantUML_Language_Reference_Guide_en.pdf"
    ]
}
"""
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate, LLMChain, HuggingFacePipeline
from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer
import torch
import os

app = Flask(__name__)

#Initialize model, tokenizer, pipeline and llm
model_id = "google/flan-t5-base" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    task = "text2text-generation",
    model=model_id,
    tokenizer=tokenizer,
    max_new_tokens=512,
    device_map="auto",
)
print("Model Loaded Successfully")
llm = HuggingFacePipeline(pipeline=pipe)

#Initialize embeddings and vectortore
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
id_dict = dict()

#Initialize text splitter 
text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=75,
      chunk_overlap=0,
      separators=[" ","\n","\n\n",".",","],
)

#Initialize llm_chain with prompt template
prompt_template = """
Context - {context}
Answer the question based on the context. {question}
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],  # List of input variables
    template=prompt_template
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

#query function
def ask_llm(question):
  similar_doc = vectorstore.similarity_search(question,k=1)
  context = similar_doc[0].page_content
  output = llm_chain({"context": context, "question": question})
  return output['text']

#delete function
def delete_file(filepaths):
  for path in filepaths:
    ids = id_dict[path]
    vectorstore.delete(ids)
    print("File removed from vectorstore!")
  return True
  
#Upload file endpoint
@app.route('/upload', methods=['POST'])
def upload_docs():
  try:
    data = request.get_json()

    if "filepath" not in data:
      return jsonify({"error":"JSON is invalid"}), 400

    if not data.get("filepath"):
      return jsonify({"error":"No file found"}), 401

    filepaths = data.get("filepath",[])

    for path in filepaths:
      loader = PDFPlumberLoader(path)
      document = loader.load()

      texts = text_splitter.split_documents(document)
      ids = vectorstore.add_documents(documents=document)
      id_dict[path] = ids
      print("Document successfully uploaded to vectorstore!")

    return jsonify({"message":"Document uploaded sucessfully"}), 200

  except Exception as e:
    return jsonify({"error": str(e)}), 500

# query endpoint
@app.route('/query', methods=['POST'])
def query_docs():
  try:
    data = request.get_json()
    query = data.get("query","")

    output = ask_llm(query)
    return jsonify({"answer":output}), 200

  except Exception as e:
    return jsonify({"error": str(e)}), 500

#delete file endpoint
@app.route('/delete', methods=['POST'])
def delete_docs():
  try:
    data = request.get_json()
    filepaths = data.get("filepath",[])

    if delete_file(filepaths):
      return jsonify({"message":"Documents deleted successfully"}), 200
    else:
      return jsonify({"error":"Documents could not be deleted"}), 400

  except Exception as e:
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()