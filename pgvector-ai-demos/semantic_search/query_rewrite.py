import json
import os
import sys
import boto3
import streamlit as st
import os
from langchain.llms.bedrock import Bedrock
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

# Add utilities to path
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utilities"))
import invoke_models

# Initialize Bedrock
bedrock_params = {
    "max_tokens_to_sample": 2048,
    "temperature": 0.0001,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\\n\\nHuman:"]
}

bedrock_region='us-west-2'
boto3_bedrock = boto3.client(service_name="bedrock-runtime", region_name=bedrock_region)
bedrock_llm = Bedrock(model_id="anthropic.claude-instant-v1", client=boto3_bedrock)
bedrock_llm.model_kwargs = bedrock_params

# Schema definition for products
schema = """{{
    "content": "Brief summary of a retail product",
    "attributes": {{
        "category": {{
            "description": "The category of the product",
            "type": "string",
            "enum": ["apparel", "footwear", "outdoors", "electronics", "beauty", "jewelry", 
                     "accessories", "housewares", "homedecor", "furniture", "seasonal", 
                     "floral", "books", "groceries", "instruments", "tools"]
        }},
        "gender_affinity": {{
            "description": "The gender that the product relates to",
            "type": "string",
            "enum": ["male", "female", "unisex"]
        }},
        "price": {{
            "description": "Cost of the product",
            "type": "double"
        }},
        "description": {{
            "description": "The detailed description of the product",
            "type": "string"
        }},
        "color": {{
            "description": "The color of the product",
            "type": "string"
        }},
        "caption": {{
            "description": "The short description of the product",
            "type": "string"
        }},
        "style": {{
            "description": "The style of the product",
            "type": "string"
        }}
    }}
}}"""

# Metadata field information
metadata_field_info = [
    AttributeInfo(
        name="price",
        description="Cost of the product",
        type="double",
    ),
    AttributeInfo(
        name="style",
        description="The style of the product",
        type="string",
    ),
    AttributeInfo(
        name="category",
        description="The category of the product",
        type="string",
    ),
    AttributeInfo(
        name="gender_affinity",
        description="The gender that the product relates to",
        type="string"
    ),
    AttributeInfo(
        name="color",
        description="The color of the product",
        type="string"
    )
]

# Few-shot examples for query rewriting
examples = [
    {
        "i": 1,
        "data_source": schema,
        "user_query": "black shoes for men",
        "structured_request": """{{
    "query": "shoes",
    "filter": "and(eq(\\"color\\", \\"black\\"), eq(\\"category\\", \\"footwear\\"), eq(\\"gender_affinity\\", \\"male\\"))"
}}"""
    },
    {
        "i": 2,
        "data_source": schema,
        "user_query": "black or brown jackets for men under 50 dollars",
        "structured_request": """{{
    "query": "jackets",
    "filter": "and(eq(\\"style\\", \\"jacket\\"), or(eq(\\"color\\", \\"brown\\"), eq(\\"color\\", \\"black\\")), eq(\\"category\\", \\"apparel\\"), eq(\\"gender_affinity\\", \\"male\\"), lt(\\"price\\", 50))"
}}"""
    },
    {
        "i": 3,
        "data_source": schema,
        "user_query": "trendy handbags for women",
        "structured_request": """{{
    "query": "handbag",
    "filter": "and(eq(\\"style\\", \\"bag\\"), eq(\\"category\\", \\"accessories\\"), eq(\\"gender_affinity\\", \\"female\\"))"
}}"""
    }
]

# Prompt templates
example_prompt = PromptTemplate(
    input_variables=['data_source', 'i', 'structured_request', 'user_query'],
    template='<< Example {i}. >>\nData Source:\n{data_source}\n\nUser Query:\n{user_query}\n\nStructured Request:\n{structured_request}\n'
)

prefix = """Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{{
    "query": string \\ text string to compare to document contents
    "filter": string \\ logical condition statement for filtering documents
}}
```

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` (eq | ne | gt | gte | lt | lte | contain | like | in | nin): comparator
- `attr` (string): name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` (and | or | not): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.
"""

suffix = """<< Example 4. >>
Data Source:
{schema}

User Query:
{query}

Structured Request:
"""

# Create few-shot prompt
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix=suffix,
    prefix=prefix,
    input_variables=["query", "schema"],
)

def parse_structured_query(structured_query_str):
    """
    Parse the structured query from LLM into pgvector-compatible format
    """
    # Parse the filter string into conditions
    conditions = {
        'must': [],
        'should': [],
        'must_not': []
    }
    
    # Extract comparisons from the filter string
    if 'filter' in structured_query_str and structured_query_str['filter'] != 'NO_FILTER':
        filter_str = structured_query_str['filter']
        
        # Simple parser for the filter conditions
        # This is a simplified version - in production, use a proper parser
        import re
        
        # Extract all eq() conditions
        eq_pattern = r'eq\("([^"]+)",\s*"([^"]+)"\)'
        for match in re.finditer(eq_pattern, filter_str):
            field, value = match.groups()
            conditions['must'].append({
                'field': field,
                'value': value,
                'operator': 'eq'
            })
        
        # Extract lt/gt conditions
        comp_pattern = r'(lt|gt|lte|gte)\("([^"]+)",\s*([^)]+)\)'
        for match in re.finditer(comp_pattern, filter_str):
            op, field, value = match.groups()
            conditions['must'].append({
                'field': field,
                'value': float(value) if field == 'price' else value,
                'operator': op
            })
        
        # Extract or conditions
        or_pattern = r'or\(([^)]+)\)'
        for match in re.finditer(or_pattern, filter_str):
            or_content = match.group(1)
            # Parse inner conditions
            for inner_match in re.finditer(eq_pattern, or_content):
                field, value = inner_match.groups()
                conditions['should'].append({
                    'field': field,
                    'value': value,
                    'operator': 'eq'
                })
    
    return structured_query_str.get('query', ''), conditions

def get_new_query_res(query):
    """
    Rewrite query using LLM to extract structured filters
    """
    # Field mapping for selected must fields
    field_map = {
        'Price': 'price',
        'Gender': 'gender_affinity',
        'Category': 'category',
        'Style': 'style',
        'Color': 'color'
    }
    
    field_map_filter = {
        key: field_map[key] 
        for key in st.session_state.get('input_must', ['Category'])
    }
    
    if not query:
        query = st.session_state.get('input_rekog_label', '')
    
    if st.session_state.get('input_is_rewrite_query') == 'enabled':
        try:
            # Get LLM response
            llm_response = invoke_models.invoke_llm_model(
                prompt.format(query=query, schema=schema),
                False
            )
            
            # Extract JSON from response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                # Clean up escaped quotes
                json_str = json_str.replace('\\"', '"')
                structured_query = json.loads(json_str)
                
                # Parse structured query
                main_query, conditions = parse_structured_query(structured_query)
                
                # Build pgvector-compatible query structure
                draft_new_query = {
                    'bool': {
                        'should': [],
                        'must': []
                    }
                }
                
                # Add main query text
                if main_query:
                    draft_new_query['bool']['must'].append({
                        'multi_match': {
                            'query': main_query.strip(),
                            'fields': ['description', 'style', 'caption']
                        }
                    })
                
                # Process conditions
                for condition in conditions['must']:
                    field = condition['field']
                    value = condition['value']
                    op = condition['operator']
                    
                    # Check if field is in must filters
                    if field in field_map_filter.values():
                        if op == 'eq':
                            draft_new_query['bool']['must'].append({
                                'match': {field: value}
                            })
                        elif op in ['lt', 'lte', 'gt', 'gte']:
                            range_query = {'range': {field: {}}}
                            range_query['range'][field][op] = value
                            draft_new_query['bool']['must'].append(range_query)
                    else:
                        # Add to should conditions
                        if op == 'eq':
                            draft_new_query['bool']['should'].append({
                                'match': {field: value}
                            })
                
                # Process should conditions
                for condition in conditions['should']:
                    field = condition['field']
                    value = condition['value']
                    
                    draft_new_query['bool']['should'].append({
                        'match': {field: value}
                    })
                
                # Store the rewritten query
                st.session_state.input_rewritten_query = {'query': draft_new_query}
                
                print("Rewritten query:")
                print(json.dumps(st.session_state.input_rewritten_query, indent=2))
                
        except Exception as e:
            print(f"Query rewriting error: {e}")
            st.session_state.input_rewritten_query = ""
