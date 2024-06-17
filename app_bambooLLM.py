import streamlit as st
import pandas as pd
import os
from pandasai.llm import AzureOpenAI
from pandasai import SmartDataframe

import os

os.environ['PANDASAI_API_KEY'] = "PANDASAI_KEY"

def chat_with_csv(df, prompt):
    pandas_ai = SmartDataframe(df)
    result = pandas_ai.chat(prompt)
    print(result)
    return result

st.set_page_config(layout='wide')
st.title("Chat with your CSV powered by LLM")

input_csv = st.file_uploader("Upload your CSV file", type=["csv"])

if input_csv is not None:
    
    col1, col2 = st.columns([1,1])

    with col1:
        st.info("CSV uploaded successfully")
        data = pd.read_csv(input_csv)
        st.dataframe(data)
    with col2:
        st.info("Chat with your CSV")

        input_text = st.text_area("Enter your query")
        if input_text is not None:
            if st.button("Ask the model!"):
                st.info("Your query: "+input_text)
                result = chat_with_csv(data, input_text)
                st.success(result)
    
