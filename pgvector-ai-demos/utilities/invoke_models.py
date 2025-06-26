import boto3
import json
#from IPython.display import clear_output, display, display_markdown, Markdown
import pandas as pd 
#from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
#from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat
import streamlit as st

#from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
#import torch

region = 'us-east-1'

def get_bedrock_client():
    return boto3.client(
    'bedrock-runtime',
    aws_access_key_id=st.secrets['user_access_key_us_west_2'],
    aws_secret_access_key=st.secrets['user_secret_key_us_west_2'], region_name = 'us-west-2'
    )


bedrock_runtime_client = get_bedrock_client()



def invoke_model(input):
    response = bedrock_runtime_client.invoke_model(
        body=json.dumps({
            'inputText': input
        }),
        modelId="amazon.titan-embed-text-v1",
        accept="application/json",
        contentType="application/json",
    )
    
    response_body = json.loads(response.get("body").read())
    return response_body.get("embedding")

def invoke_model_mm(text,img):
    body_ = {
            "inputText": text,
            
        }
    if(img!='none'):
        body_['inputImage']=img

    body = json.dumps(body_)
        
    modelId = 'amazon.titan-embed-image-v1'
    accept = 'application/json'
    contentType = "application/json"

    response = bedrock_runtime_client.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
    response_body = json.loads(response.get("body").read())
    #print(response_body)
    return response_body.get("embedding")

def invoke_llm_model(input,is_stream):
    if(is_stream == False):
        response = bedrock_runtime_client.invoke_model( 
            modelId= "anthropic.claude-3-haiku-20240307-v1:0",#"anthropic.claude-3-5-sonnet-20240620-v1:0",,
            contentType = "application/json",
            accept = "application/json",
   
            body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1024,
                        "temperature": 0.001,
                        "top_k": 250,
                        "top_p": 1,
                        "stop_sequences": [
                            "\n\nHuman:"
                        ],
                        "messages": [
                        {
                            "role": "user",
                            "content":input
                            }
                            ]
                        }
                        
                         )
            )
        
        res = (response.get('body').read()).decode()
        
        return (json.loads(res))['content'][0]['text']
        
      
def read_from_table(file,question):
    print("started table analysis:")
    print("-----------------------")
    print("\n\n")
    print("Table name: "+file)
    print("-----------------------")
    print("\n\n")
    bedrock_params = {
    "max_tokens":2048,
    "temperature":0.0001,
    "top_k":150,
    "top_p":0.7,
    "stop_sequences":["\\n\\nHuman:"] 
    }
    
    model = BedrockChat(
    client=bedrock_runtime_client,
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
    model_kwargs=bedrock_params,
    streaming=False
    )
    if(str(type(file))=="<class 'str'>"):
        df = pd.read_csv(file,skipinitialspace = True, on_bad_lines='skip',delimiter = "`")
    else:
        df = file
    agent = create_pandas_dataframe_agent(
             model, 
             df, 
             verbose=True,
             agent_executor_kwargs={'handle_parsing_errors':True,
                                    'return_only_outputs':True},allow_dangerous_code = True
             )
    agent_res = agent.invoke(question)['output']
    return agent_res
    
def generate_image_captions_llm(base64_string,question):
    
   
    response = bedrock_runtime_client.invoke_model( 
            modelId= "anthropic.claude-3-haiku-20240307-v1:0",
            contentType = "application/json",
            accept = "application/json",
   
            body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1024,
                        "messages": [
                        {
                            "role": "user",
                            "content": [
                            {
                                "type": "image",
                                "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_string
                                }
                            },
                            {
                                "type": "text",
                                "text": question
                            }
                            ]
                        }
                        ]
                         }))
    response_body = json.loads(response.get("body").read())['content'][0]['text']
    return response_body