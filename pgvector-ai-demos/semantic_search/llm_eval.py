import os
import sys
import json
import numpy as np
import streamlit as st
import os
from sklearn.metrics import ndcg_score, dcg_score
from sklearn import preprocessing as pre

# Add utilities to path
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utilities"))
import invoke_models as llm

def eval(questions, answers):
    """
    Evaluate search results using LLM to assess relevance
    """
    if not questions or not answers or not answers[0]['answer']:
        return
    
    # Prepare evaluation prompt
    prompt = """Human: You are a grader assessing relevance of retrieved documents to a user question.
    The User question and Retrieved documents are provided below. The Retrieved documents are retail product descriptions that the human is looking for.
    It does not need to be a stringent test. The goal is to filter out totally irrelevant product retrievals.
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    
    <User question>
    {}
    </User question>

    <Retrieved documents>
    {}
    </Retrieved documents>

    Now based on the information provided above, for every given Retrieved document, provide:
    - The index of the document
    - A relevance score out of 5 based on relevance with the User question
    - Whether it is relevant (true/false)
    - Reason why it is relevant or not
    
    Provide your response as a JSON object with the document index as keys.
    
    Answer:
    """
    
    # Extract query and results
    query = questions[0]['question']
    search_results = ""
    
    for idx, result in enumerate(answers[0]['answer']):
        desc = f"{result.get('caption', '')}. {result.get('desc', '')}"
        search_results += f"Index: {idx}, Description: {desc}\n\n"
    
    # Get LLM evaluation
    full_prompt = prompt.format(query, search_results)
    
    try:
        response_text = llm.invoke_llm_model(full_prompt, False)
        
        # Parse JSON response
        # Try to extract JSON from the response
        if '{' in response_text and '}' in response_text:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            response = json.loads(json_str)
        else:
            # Fallback: create default response
            response = {str(i): {"relevant": True, "score": 3} 
                       for i in range(len(answers[0]['answer']))}
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        # Fallback: assume all results are somewhat relevant
        response = {str(i): {"relevant": True, "score": 3} 
                   for i in range(len(answers[0]['answer']))}
    
    # Extract scores and update results
    llm_scores = []
    current_scores = []
    
    for idx, result in enumerate(answers[0]['answer']):
        idx_str = str(idx)
        
        if idx_str in response:
            relevance = response[idx_str].get('relevant', True)
            score = response[idx_str].get('score', 3)
        else:
            relevance = True
            score = 3
        
        result['relevant'] = relevance
        llm_scores.append(score)
        current_scores.append(result.get('score', 0))
    
    # Calculate NDCG
    if llm_scores and any(s > 0 for s in llm_scores):
        # Normalize scores
        x = np.array(llm_scores).reshape(-1, 1)
        x_norm = pre.MinMaxScaler().fit_transform(x).flatten()
        
        y = np.array(current_scores).reshape(-1, 1)
        y_norm = pre.MinMaxScaler().fit_transform(y).flatten()
        
        # Calculate DCG
        dcg = dcg_score(np.asarray([llm_scores]), np.asarray([current_scores]))
        
        # Calculate IDCG (ideal DCG)
        idcg = dcg_score(np.asarray([llm_scores]), np.asarray([llm_scores]))
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        
        print(f"NDCG: {ndcg}")
        
        # Update session state
        previous_ndcg = st.session_state.get('input_ndcg', 0)
        
        if ndcg > previous_ndcg and previous_ndcg != 0:
            st.session_state.ndcg_increase = f"↑~{ndcg - previous_ndcg:.3f}"
        elif ndcg < previous_ndcg:
            st.session_state.ndcg_increase = f"↓~{previous_ndcg - ndcg:.3f}"
        else:
            st.session_state.ndcg_increase = " ~ "
        
        st.session_state.input_ndcg = ndcg
    else:
        # No valid scores
        st.session_state.input_ndcg = 0
        st.session_state.ndcg_increase = " ~ "
    
    # Update answers in session state
    st.session_state.answers = answers
