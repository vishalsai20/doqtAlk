import streamlit as st
# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from dotenv import load_dotenv

# load_dotenv()
api_key = st.secrets["PINECONE_API_KEY"]

# =========================
# Helper functions
# =========================
def download_embeddings():
    model = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model)

def clean_answer_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = line.replace("**", "")
        if line.startswith("*"):
            line = "- " + line.lstrip("* ").strip()
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

# =========================
# Cache heavy objects
# =========================
@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    embeddings = download_embeddings()

    index_name = "check"  # your Pinecone index
    docsearch = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    model = GoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.0,
        max_output_tokens=256
    )

    system_prompt = (
        "You are a Medical Expert for question-answering tasks. "
        "Use the retrieved context to answer the query. "
        "If you don't know the answer, say you don't know. "
        "Do not hallucinate. "
        "Keep the answer concise and well structured.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    qna_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, qna_chain)

    return rag_chain

# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="DoqtAlk",
    layout="centered"
)

st.title("Medical Assistant")
st.write("Replies sourced from 'The Gale Encyclopedia of Medicine'.")

# Load pipeline once
rag_chain = load_rag_pipeline()

# User input
user_query = st.text_area(
    "Post your query:",
    placeholder="Typing....",
    height=100
)

# Submit button
if st.button("Submit"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response.get("answer", "")
            cleaned_answer = clean_answer_text(answer)

        st.subheader("Answer:")
        st.write(cleaned_answer)
