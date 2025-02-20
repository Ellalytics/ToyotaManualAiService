import os
import requests
import streamlit as st
from tempfile import NamedTemporaryFile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

# Suppress Warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

##############################################################################################################
################################### Function for retrieving the vector store ###################################
##############################################################################################################
def build_vector_store(content):
    if content:
        # If the vector store is not already present in the session state
        if not st.session_state.vector_store:

            with st.spinner(text=":red[Please wait while we fetch the information...]"):

                ################################# Fetch the embedding file ##################################
                embedding = HuggingFaceEmbeddings()
                embedding_file = 'https://raw.githubusercontent.com/Samuelchazy/Educative.io/8f0e764c1b69e2d61f4e44e3084c0695d85cd6e8/persistence/user_manuel.json'

                # Download the embedding file from the URL and save it temporarily
                with NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
                    response = requests.get(embedding_file)
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name

                vector_store = SKLearnVectorStore(embedding=embedding,
                                                    persist_path=tmp_file_path,
                                                    serializer='json')

                ######################### Save the vector store to the session state ########################
                st.session_state.vector_store = vector_store
                return vector_store

        else:
            # Load the vector store from the cache
            return st.session_state.vector_store

    else:
        st.error('No content was found...')

##############################################################################################################
###################### Function for retrieving the relevant chunks from the vector store #####################
##############################################################################################################
def retrieve_chunks_from_vector_store(vector_store, re_written_query):

    ########################### Perform a similarity search with relevance scores ############################
    with st.spinner(text=":red[Please wait while we fetch the relevant information...]"):
        relevant_documents = vector_store.similarity_search_with_score(query=re_written_query, k=5)
        return relevant_documents

##############################################################################################################
################################### Function for retrieving the chat history #################################
##############################################################################################################
def retrieve_history():
    ############################## Go through all the chat messages in the history ###########################
    for message in st.session_state.messages:
        with st.container(border=True):
            with st.chat_message(message['role']):
                st.markdown(message['content'])
