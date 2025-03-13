import os
from unicodedata import category

import requests
import streamlit as st
from tempfile import NamedTemporaryFile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore, FAISS
from sentence_transformers import SentenceTransformer

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
    try:
        # Ensure content is a list of strings
        if isinstance(content, str):
            content = [content]
        elif not isinstance(content, list):
            content = list(map(str, content))

            # Validate non-empty content after conversion
        if not content:
            st.error("No valid content to create vector store.")
            return None

            # Detailed logging and error checking
        print(f"Content type: {type(content)}")
        print(f"Number of documents: {len(content)}")
        print(f"Sample content: {content[:2]}")  # Show first two items
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Use HuggingFace embeddings with error handling
        try:
            embedding = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",  # Specify a specific model
                model_kwargs={'device': 'cpu'},  # Explicitly set device
                encode_kwargs={'normalize_embeddings': True}
            )

        except Exception as embed_error:
            st.error(f"Embedding model initialization error: {embed_error}")
            return None

        # Validate embedding
        try:
            test_embedding = embedding.embed_documents(content[:1])
            st.write(f"Embedding dimension: {len(test_embedding[0])}")
        except Exception as embed_test_error:
            st.error(f"Embedding test failed: {embed_test_error}")
            return None

        # Create vector store using FAISS with spinner
        with st.spinner(":red[Processing content, please wait...]"):
            try:
                vector_store = FAISS.from_texts(
                    texts=content,
                    embedding=embedding
                )

                # Save to session state
                st.session_state.vector_store = vector_store

                st.success("Vector store created successfully!")
                return vector_store

            except Exception as faiss_error:
                st.error(f"FAISS vector store creation error: {faiss_error}")
                return None

    except Exception as e:
        st.error(f"Unexpected error in vector store creation: {e}")
        return None


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
