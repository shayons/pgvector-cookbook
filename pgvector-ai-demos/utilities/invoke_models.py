"""
Updated invoke_models.py for HF Spaces deployment with environment variables
"""
import boto3
import json
import os
import pandas as pd 
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.chat_models import BedrockChat
import streamlit as st

region = 'us-east-1'

def get_bedrock_client(region_name='us-west-2'):
    """Get Bedrock client using environment variables"""
    if region_name == 'us-west-2':
        return boto3.client(
            'bedrock-runtime',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID_US_WEST_2', os.environ.get('AWS_ACCESS_KEY_ID')),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY_US_WEST_2', os.environ.get('AWS_SECRET_ACCESS_KEY')),
            region_name='us-west-2'
        )
    else:
        return boto3.client(
            'bedrock-runtime',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=region_name
        )

def get_sagemaker_client():
    """Get SageMaker runtime client"""
    return boto3.client(
        'sagemaker-runtime',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )

# Initialize clients
bedrock_runtime_client = get_bedrock_client('us-west-2')
sagemaker_runtime = get_sagemaker_client()

def invoke_model(input_text):
    """Generate text embeddings using Amazon Titan"""
    try:
        response = bedrock_runtime_client.invoke_model(
            body=json.dumps({
                'inputText': input_text
            }),
            modelId="amazon.titan-embed-text-v1",
            accept="application/json",
            contentType="application/json",
        )
        
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding")
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def invoke_model_mm(text, img):
    """Generate multimodal embeddings using Amazon Titan"""
    try:
        body_ = {
            "inputText": text,
        }
        if img != 'none':
            body_['inputImage'] = img

        body = json.dumps(body_)
        modelId = 'amazon.titan-embed-image-v1'
        accept = 'application/json'
        contentType = "application/json"

        response = bedrock_runtime_client.invoke_model(
            body=body, 
            modelId=modelId, 
            accept=accept, 
            contentType=contentType
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding")
    except Exception as e:
        st.error(f"Error generating multimodal embedding: {e}")
        return None

def invoke_llm_model(input_text, is_stream):
    """Generate text using Claude"""
    try:
        if not is_stream:
            response = bedrock_runtime_client.invoke_model( 
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "temperature": 0.001,
                    "top_k": 250,
                    "top_p": 1,
                    "stop_sequences": ["\n\nHuman:"],
                    "messages": [{
                        "role": "user",
                        "content": input_text
                    }]
                })
            )
            
            res = (response.get('body').read()).decode()
            return (json.loads(res))['content'][0]['text']
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return "Sorry, I encountered an error processing your request."

def read_from_table(file, question):
    """Analyze table data using pandas agent and Claude"""
    try:
        print("Started table analysis:")
        print("-----------------------")
        print(f"Table name: {file}")
        print("-----------------------")
        
        bedrock_params = {
            "max_tokens": 2048,
            "temperature": 0.0001,
            "top_k": 150,
            "top_p": 0.7,
            "stop_sequences": ["\\n\\nHuman:"] 
        }
        
        model = BedrockChat(
            client=bedrock_runtime_client,
            model_id='anthropic.claude-3-haiku-20240307-v1:0',
            model_kwargs=bedrock_params,
            streaming=False
        )
        
        if str(type(file)) == "<class 'str'>":
            df = pd.read_csv(file, skipinitialspace=True, on_bad_lines='skip', delimiter="`")
        else:
            df = file
            
        agent = create_pandas_dataframe_agent(
            model, 
            df, 
            verbose=True,
            agent_executor_kwargs={'handle_parsing_errors': True, 'return_only_outputs': True},
            allow_dangerous_code=True
        )
        agent_res = agent.invoke(question)['output']
        return agent_res
    except Exception as e:
        st.error(f"Error analyzing table: {e}")
        return "Unable to analyze table data."
    
def generate_image_captions_llm(base64_string, question):
    """Generate image captions using Claude Vision"""
    try:
        response = bedrock_runtime_client.invoke_model( 
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_string
                        }
                    }, {
                        "type": "text",
                        "text": question
                    }]
                }]
            })
        )
        response_body = json.loads(response.get("body").read())['content'][0]['text']
        return response_body
    except Exception as e:
        st.error(f"Error generating image caption: {e}")
        return "Unable to analyze image."

def invoke_sagemaker_endpoint(endpoint_name, payload):
    """Invoke SageMaker endpoint for custom models"""
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        result = json.loads(response['Body'].read().decode())
        return result
    except Exception as e:
        st.error(f"Error calling SageMaker endpoint {endpoint_name}: {e}")
        return None