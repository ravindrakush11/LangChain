import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()

os.environ['LANGSMITH_TRACING'] = 'true'

import getpass
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your Langsmith API key(Optional): ")
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass("Enter your Langsmith project name(optional): ") or 'default'
if "GROQ_API_KEY" not in os.environ:
    os.environ['GROQ_API_KEY'] = getpass.getpass("Enter your GROQ API key (required if using GROQ): ")

model = init_chat_model('llama3-8b-8192', model_provider='groq')

app = Flask(__name__)

system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

@app.route('/translate', methods=['POST'])
def translate():
    """
    Input JSON:
    {
        "text": "Hi, how are you?,
        "language: "Hindi"
    }
    """
    data = request.get_json()
    text = data.get('text')
    language = data.get('language', 'Hindi')

    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    prompt = prompt_template.invoke({
        "language": language,
        "text": text
    })
    response = model.invoke(prompt)
    return jsonify({
        "translate_text": response.content 
    })

@app.route('/')
def home():
    return "LangChain Translation API is running!"

if __name__ == '__main__':
    app.run(debug=True)

