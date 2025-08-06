import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Chat with CSV", layout="wide")
st.title("📊 Chat with your CSV using AI")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### CSV Preview", df.head())

        # Clean object (string) columns
        clean_df = df.select_dtypes(include=["object"]).copy()

        # Drop long text columns (>200 avg chars)
        for col in clean_df.columns:
            if clean_df[col].astype(str).str.len().mean() > 200:
                clean_df.drop(columns=[col], inplace=True)

        # Limit to 20 columns (optional safety)
        clean_df = clean_df.iloc[:, :20]

        if openai_api_key:
            try:
                llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
                llm.invoke("ping")  # force a quick test
            except Exception as e:
                st.warning("Falling back to gpt-3.5-turbo (gpt-4 not available).")
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

            try:
                agent = create_pandas_dataframe_agent(
                            llm,
                            clean_df,
                            verbose=False,
                            allow_dangerous_code=True
                        )

                # Chat UI
                st.write("### Ask a question about your CSV:")
                user_question = st.text_input("Enter your question here")

                if user_question:
                    with st.spinner("Thinking..."):
                        answer = agent.run(user_question)
                        st.success(answer)

            except ValueError as e:
                st.error("Agent creation failed.")
                st.code(str(e))

        else:
            st.warning("Missing OpenAI API key. Add it to your .env or Streamlit secrets.")

    except Exception as e:
        st.error("Could not read CSV file.")
        st.code(str(e))
