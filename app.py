import io
import hashlib
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

st.set_page_config(
    page_title="Employee Onboarding AI Assistant",
    page_icon="🤖"
)

st.title("🤖 Employee Onboarding AI Assistant")
st.write("Upload an HR policy PDF or ask a general question.")

uploaded_file = st.file_uploader(
    "Upload HR Policy PDF (optional)",
    type=["pdf"]
)

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOllama(model="llama3.2", temperature=0)

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_text, doc_id):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = splitter.create_documents([pdf_text])

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="hr_" + doc_id
    )

pdf_text = ""
vectorstore = None

if uploaded_file is not None:
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        pdf_text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

        if not pdf_text.strip():
            st.error("This PDF has no readable text. Try another PDF.")
            st.stop()

        doc_id = hashlib.sha256(
            pdf_text.encode("utf-8")
        ).hexdigest()[:16]

        with st.spinner("Reading PDF..."):
            vectorstore = build_vectorstore(pdf_text, doc_id)

        st.success("PDF uploaded and processed successfully!")

    except Exception as error:
        st.error("Could not read the PDF.")
        st.code(str(error))
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Type your question here...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Preparing answer..."):
                if vectorstore is not None:
                    matches = vectorstore.similarity_search(
                        question, k=4
                    )
                    context = "\n\n".join(
                        doc.page_content for doc in matches
                    )

                    prompt = f"""
You are an Employee Onboarding AI Assistant.
Answer using ONLY the uploaded PDF context.
If the answer is not in the PDF, tell the user to contact HR.
Do not invent company policies.

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
                else:
                    prompt = f"""
You are a helpful AI assistant.
Answer the user's general question clearly.
Do not claim that you have read any company policy.

QUESTION:
{question}

ANSWER:
"""

                response = get_llm().invoke(prompt)
                answer = response.content

            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as error:
            answer = (
                "Could not generate an answer. "
                "Please check whether Ollama is running."
            )
            st.error(answer)
            st.code(str(error))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )