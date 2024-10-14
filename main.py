import streamlit as st
from chromadb import Client
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.callbacks.base import BaseCallbackHandler
from huggingface_hub import hf_hub_download
import os

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

@st.cache_resource
def create_chain(system_prompt):
    (repo_id, model_file_name) = ("TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                                  "mistral-7b-instruct-v0.1.Q4_0.gguf")

    model_path = hf_hub_download(repo_id=repo_id,
                                 filename=model_file_name,
                                 repo_type="model")

    llm = LlamaCpp(
            model_path=model_path,
            temperature=0,
            max_tokens=512,
            top_p=1,
            stop=["[INST]"],
            verbose=False,
            streaming=True,
            )

    template = """
    <s>[INST]{}[/INST]</s>

    [INST]{}[/INST]
    """.format(system_prompt, "{question}")

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = prompt | llm

    return llm_chain

def load_data(file_path):
    with open(file_path, "r") as file:
        data = file.readlines()
    return [line.strip() for line in data if line.strip()]

def create_vector_store(data):
    client = Client()
    tenant_name = "default_tenant"
    try:
        client.set_tenant(tenant_name)
    except ValueError:
        client.create_tenant(name=tenant_name)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(data, embeddings)
    
    return vector_store

def get_similar_chunks(user_prompt, vector_store):
    similar_docs = vector_store.similarity_search(user_prompt, k=3)
    return " ".join([doc.page_content for doc in similar_docs])

st.set_page_config(page_title="Faktus.eu")
st.header("Faktus.eu")

system_prompt ="Tu es un assistant de Faktus tu réponds clairement en français.",


data_file = "data.txt"
if os.path.exists(data_file):
    data_chunks = load_data(data_file)
    vector_store = create_vector_store(data_chunks)
else:
    st.error("data.txt file not found.")

llm_chain = create_chain(system_prompt)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Comment puis-je vous aider?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Votre message", key="user_input"):
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    context = get_similar_chunks(user_prompt, vector_store)
    full_prompt = f"{context}\n\n{user_prompt}. Réponds en français."
    response = llm_chain.invoke({"question": full_prompt})

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)
