from flask import Flask, request
from langchain import PromptTemplate, LLMChain, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
import jsonify

app = Flask(__name__)

model = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

falcon_pipeline = pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    do_sample=True,
    max_new_tokens=500,
    top_p=0.15,
    return_full_text=True,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

@app.route('/falcon', methods=["POST"])
def generate_json():
    try:
        data = request.get_json()

        if "name" not in data or "name" not in data.values():
            return jsonify({"error": "Invalid JSON data"}), 400
        
        form_name = data['name']
        
        llm = HuggingFacePipeline(pipeline= falcon_pipeline, model_kwargs={'temperature':0})
        
        template = """
        Create a JSON packet for field names in {form_name} form with label, type, required, default value.
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


    