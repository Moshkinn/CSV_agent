import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent

# Load environment variables (for OpenAI key)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up Streamlit app UI
st.set_page_config(page_title="Chat with CSV", layout="wide")
st.title("📊 Chat with your CSV using AI")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your data:", df.head())

    # Initialize LangChain agent
    if openai_api_key:
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
        agent = create_pandas_dataframe_agent(llm, df, verbose=False)

        # Chat interface
        st.write("### Ask questions about your data:")
        user_question = st.text_input("Enter your question")

        if user_question:
            with st.spinner("Thinking..."):
                try:
                    response = agent.run(user_question)
                    st.success(response)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
