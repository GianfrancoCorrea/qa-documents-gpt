from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.document_loaders import TextLoader
import pickle
import os
import streamlit as st 
import configparser
import os

config = configparser.ConfigParser()
config.read('config.ini')
assistant_api_key = config['DEFAULT']['API-KEY']


os.environ['OPENAI_API_KEY'] = assistant_api_key
embeddings = OpenAIEmbeddings()

def save(db):
  with open("state_of_the_union.vectorstore.pkl", "wb") as f:
	    pickle.dump(db, f)

def load():
  with open("./state_of_the_union.vectorstore.pkl", "rb") as f:
      db = pickle.load(f)
      return db

def create_db():
    loader = TextLoader('./state_of_the_union.txt', encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)
    save(db)

    return db

# Example usage:
# db = create_db()

def get_docs_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    return docs

def get_response_from_query(query, docs):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """
    docs_page_content = " ".join([d.page_content for d in docs])
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

    # Template to use for the system message prompt
    template = """
        This is a document: {docs}
        
        answer about that
        
        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response

db = load()
original_docs = "./state_of_the_union.txt"

# App framework
st.title('🦜 QA from documents with VectorDB 🔗')
# upload txt file (just txt files for now)

uploaded_file = st.file_uploader("Choose a file (optional)", type=['txt'])

if uploaded_file is not None:
    button = st.button('Create VectorDB')
    if button:
        # create vector db
        st.info('Creating VectorDB...')
        loader = TextLoader(uploaded_file.name, encoding='utf-8') # TODO: fix this
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        db = FAISS.from_documents(docs, embeddings)

        # replace vector db
        db = load()
        original_docs = uploaded_file.name

        st.info('Done creating VectorDB, use the prompt below to query the db for answers.')
        with st.expander('Loaded txt'): 
            with open(uploaded_file.name, 'r', encoding='utf-8') as f:
                text = f.read()
                st.info(text)

# add separator
st.markdown("---")


# required prompt input
prompt = st.text_input('Plug in your prompt here')

if prompt:
    docs = get_docs_from_query(db, prompt)
    response = get_response_from_query(prompt, docs)
    st.write(response)

    with st.expander('Docs from VectorDB'): 
        st.info(docs)
    with st.expander('Original docs: ' + original_docs.split('/')[-1]):
        with open(original_docs, 'r', encoding='utf-8') as f:
            text = f.read()
            st.info(text)