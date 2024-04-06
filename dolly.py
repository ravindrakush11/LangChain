from flask import Flask, request
from langchain import PromptTemplate, LLMChain, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
import jsonify
import os 

app = Flask(__name__)

model = "databricks/dolly-v2-3b"

dolly_pipeline = pipeline(
    "text-generation", 
    model=model,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    return_full_text=True,
)

@app.route('/dolly', methods=["POST"])
def generate_json():
    try:
        data = request.get_json()

        if "name" not in data or "name" not in data.values():
            return jsonify({"error": "Invalid JSON data"}), 400
        
        form_name = data['name']
        
        llm = HuggingFacePipeline(pipeline= dolly_pipeline)
        
        template = """
        Create a JSON packet for field {form_name} form fields with label, type, required, default value.
        Dont produce any extra text.
        """
        prompt = PromptTemplate(template=template, input_variables=['form_name'])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        output = llm_chain.run(form_name)
        return jsonify({"output": str(output)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)