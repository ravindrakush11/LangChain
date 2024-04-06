import os
import re
import PyPDF2
import spacy
import pandas as pd
from spacy.matcher import Matcher
from tqdm import tqdm
from flask import Flask, request, jsonify

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    cleaned_text = re.sub(r'/["\n\s\W]+/g', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def split_into_sentences(document):
    doc = nlp(document)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def prepare_data(folder_path):
    pdf_texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # file must be pdf
        if filename.lower().endswith(".pdf"):
            try:
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = ""

                    # Iterate through each page in the PDF and extract text
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text()

                    text = preprocess_text(text)
                    text = split_into_sentences(text)
                    pdf_texts.append(text)
            except Exception as e:
                print(str(e))

    return pdf_texts

def get_entities(sent):
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  
    prv_tok_text = ""  

    prefix = ""
    modifier = ""

    for tok in nlp(sent):
    
        if tok.dep_ != "punct":
            
            if tok.dep_ == "compound":
                prefix = tok.text

                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text

    return [ent1.strip(), ent2.strip()]

def get_relation(sent):

    doc = nlp(sent)

    matcher = Matcher(nlp.vocab)

    pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}]

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]

    return(span.text)

def create_triplets(pdf_texts):
    entity_pairs = []
    relations = []

    for doc in pdf_texts:
        for sent in tqdm(doc):
            entity_pairs.append(get_entities(sent))
            relations.append(get_relation(sent))

    source = [i[0] for i in entity_pairs]
    target = [i[1] for i in entity_pairs]

    #kg_df = dict({'source': source, 'target': target, 'edge':relations})
    kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})
    print(kg_df.tail(25))
    print("triplets are created")
    return kg_df

def get_output(kg_df):
    node_list = []
    edge_list = []

    unique_labels = set(kg_df['source'].unique()).union(kg_df['target'].unique())

    node_list = [{'node_id': idx, 'label': label} for idx, label in enumerate(unique_labels)]

    for i, row in kg_df.iterrows():
        edge = {
            'edge_id': len(node_list)+i,  # Adjust the range as needed
            'label': row['edge'],  # You can use the "i-th relation" here
            'source': [node['node_id'] for node in node_list if node['label'] == row['source']][0],
            'target': [node['node_id'] for node in node_list if node['label'] == row['target']][0]
        }
        edge_list.append(edge)

    return node_list, edge_list


@app.route("/upload", methods=["POST"])
def create_kg():
    try:
        data = request.get_json()
        kg_dict = dict()
        
        if not data.get('folder') or 'folder' not in data:
            return jsonify({"error": "Invalid JSON data"}), 400
        
        folder_path = data['folder']

        if not os.path.exists(folder_path):
            return jsonify({"error": "Folder path does not exist"}), 401
        
        pdf_texts = prepare_data(folder_path)

        if not pdf_texts:
            return jsonify({"error": "Unable to read pdf data"}), 402

        kg_df = create_triplets(pdf_texts)

        if kg_df.empty:
            return jsonify({"error": "Unable to extract triplets"}), 403
        
        #Uncomment code to get tripletes as output
        # kg_dict = kg_df.to_dict(orient='records')
        # return jsonify(kg_df), 200

        node_list, edge_list = get_output(kg_df)

        return jsonify({"Node":node_list, "Edge": edge_list}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run()