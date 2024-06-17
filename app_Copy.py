import streamlit as st
import pandas as pd
import os
from pandasai.llm import AzureOpenAI
from pandasai import SmartDataframe


# Access environment variables
api_token = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_ENDPOINT')
api_version = os.getenv('API_VERSION')
deployment_name = os.getenv('DEPLOYMENT_NAME')

# Initialize the AzureOpenAI instance
azure_llm = AzureOpenAI(
    api_token=api_token,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    deployment_name=deployment_name
)

def chat_with_csv(df, prompt):
    pandas_ai = SmartDataframe(df, config={"llm": azure_llm})
    result = pandas_ai.chat(prompt)
    return result

# Set Streamlit page configuration
st.set_page_config(layout='wide')
st.title("Chat with your CSV/Excel file: Powered by LLM")

# Upload CSV file
input_csv = st.file_uploader("Upload your CSV file", type=["csv", "xlsx", "xls"])

if input_csv is not None:
    
    col1, col2 = st.columns([1,1])

    with col1:
        st.info("CSV uploaded successfully")
        data = pd.read_csv(input_csv)
        st.dataframe(data)

    with col2:
        st.info("Chat with your CSV")
        input_text = st.text_area("Enter your query")

        if input_text:
            if st.button("Ask the model!"):
                st.info("Your query: " + input_text)
                result = chat_with_csv(data, input_text)
                
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                else:
                    st.success(result)
    
    


