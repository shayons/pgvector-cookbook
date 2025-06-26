from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import streamlit as st
import os
import boto3
import json

runtime = boto3.client('sagemaker-runtime',aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),region_name='us-west-2')
# Load Tokenizer from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
endpoint_name = 'all-MiniLM-L6-v2-model'


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
           torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)


def vectorise(sentence,token_level_vectors):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    # Get input IDs (token IDs)
    input_ids = encoded_input['input_ids'][0]  

    # Convert IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    model_output = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps({"inputs": sentence})
    )
    token_vectors = json.loads(model_output['Body'].read().decode())
    if(token_level_vectors):
        return tokens,token_vectors
    
    token_vectors_tensor = torch.tensor(token_vectors) 
    attention_mask = encoded_input['attention_mask']  
    
    # Perform pooling
    sentence_embeddings = mean_pooling(token_vectors_tensor, attention_mask)

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings[0].tolist()

def search(hits):
    tokens,token_vectors = vectorise(st.session_state.input_text,True)
    final_docs = []
    for ind,j in enumerate(hits):
        max_score_dict_list = []
        doc={"_source":
            {
            "product_description":j["_source"]["description"],"caption":j["_source"]["title"],
            "image_url":j["_source"]["image_s3_url"].replace("https://retail-demo-store-us-east-1.s3.amazonaws.com/images/","/home/user/app/images_retail/")
            ,"price":j["_source"]["price"],
            "style":j["_source"]["style"],"category":j["_source"]["category"]},"_id":j["_id"],"_score":j["_score"]}
            
        if("gender_affinity" in j["_source"]):
            doc["_source"]["gender_affinity"] = j["_source"]["gender_affinity"]
        else:
            doc["_source"]["gender_affinity"] = ""
        source_doc_token_keys = list(j["_source"].keys())
        with_s = [x for x in source_doc_token_keys if x.startswith("description-token-")]
        add_score = 0
        
        for index,i in enumerate(token_vectors[0]):
            token = tokens[index]
            if(token!='[SEP]' and token!='[CLS]'):
                query_token_vector = np.array(i)
                scores = []
                for m in with_s:
                    m_arr = m.split("-")
                    if(m_arr[-1]!='[SEP]' and m_arr[-1]!='[CLS]'):
                        doc_token_vector = np.array(j["_source"][m])
                        score = np.dot(query_token_vector,doc_token_vector)
                        scores.append({"doc_token":m_arr[3],"score":score})
                        
                newlist = sorted(scores, key=lambda d: d['score'], reverse=True)
                max_score = newlist[0]['score']
                add_score+=max_score
                max_score_dict_list.append(newlist[0])
                
        max_score_dict_list_sorted = sorted(max_score_dict_list, key=lambda d: d['score'], reverse=True)
        print(max_score_dict_list_sorted)
        
        doc["total_score"] = add_score
        doc['max_score_dict_list_sorted'] = max_score_dict_list_sorted
        final_docs.append(doc)
    final_docs_sorted = sorted(final_docs, key=lambda d: d['total_score'], reverse=True)
    return final_docs_sorted
        
                
        
                
        