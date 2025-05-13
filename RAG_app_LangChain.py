from flask import Flask, request, jsonify
import os
import numpy as np
import tiktoken
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

# ========== ENVIRONMENT SETUP ==========
from dotenv import load_dotenv
load_dotenv()

# ========== INITIALIZE MODELS ==========
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ========== COSINE SIMILARITY FUNCTION ==========
def embed_text(text):
    return embeddings.embed_query(text)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return float(dot_product / (norm_vec1 * norm_vec2))

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

# ========== DOCUMENT INDEXING FOR RAG ==========
def build_retriever():
    loader = WebBaseLoader(
        web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
        ),
    )
    blog_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(blog_docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    return retriever

retriever = build_retriever()

# ========== PROMPT AND RAG CHAIN ==========
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ========== FLASK APP ==========
app = Flask(__name__)

@app.route("/similarity", methods=["POST"])
def similarity():
    data = request.get_json()
    question = data.get("question")
    document = data.get("document")

    if not question or not document:
        return jsonify({"error": "Both 'question' and 'document' are required."}), 400

    query_vec = embed_text(question)
    doc_vec = embed_text(document)
    similarity_score = cosine_similarity(query_vec, doc_vec)

    return jsonify({
        "cosine_similarity": similarity_score,
        "tokens_in_question": num_tokens_from_string(question),
        "tokens_in_document": num_tokens_from_string(document)
    })

@app.route("/rag", methods=["POST"])
def rag():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question is required."}), 400

    response = rag_chain.invoke(question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
