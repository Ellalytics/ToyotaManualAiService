import langchain_core
import requests
import streamlit as st
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import PyPDF2
import json
import os

from numpy.f2py.auxfuncs import isstring


####################### Function for loading and caching the content of the PDF file #######################
@st.cache_data
def load_pdf_content(user_manuel_path):
    # Check if the local PDF file exists
    if not os.path.exists(user_manuel_path):
        st.error(f"File '{user_manuel_path}' not found.")
        return None

    # Load the PDF content using PyPDFLoader
    pdf_loader = PyPDFLoader(user_manuel_path)
    pdf_reader = pdf_loader.load()

    # Helper function to split text into chunks
    def chunk_text(text, chunk_size=1000):
        # Split the text into chunks of the specified size
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Extract and format the content, and split into chunks
    content = []
    for page in pdf_reader:
        if not page:
            page_content = '...'
        elif isstring(page):  # Check if page is a string (though it seems it might not be in your case)
            page_content = page.replace('\n', '\n\n')
        elif isinstance(page, langchain_core.documents.base.Document):  # Ensure it's a Document object
            page_content = page.page_content.replace('\n', '\n\n')
        else:
            page_content = '...'  # Default if it's an unexpected type

        # Now you can process the page content
        chunks = chunk_text(page_content)
        content.extend(chunks)
    return content
###########################################################################################################
########################### Function for displaying the PDF file and the images ###########################
###########################################################################################################
def load_file():
    with st.spinner('Loading PDF content. Please wait around a minute...'):
        user_manuel_path = os.environ['TOYOTA_MANUAL_PATH']
        content = load_pdf_content(user_manuel_path)
    if content:
        with st.container(height=600, border=False):
            col_left, col_right = st.columns(2)
            ###################################### Display the images #####################################
            with col_left:
                image_path = "https://raw.githubusercontent.com/Samuelchazy/Educative.io/19d3100db50749489689a5c21029c3499722b254/images/Toyota_3.jpg"
                st.image(image_path, use_container_width=True)

                image_path = "https://raw.githubusercontent.com/Samuelchazy/Educative.io/19d3100db50749489689a5c21029c3499722b254/images/Toyota_4.jpg"
                st.image(image_path, use_container_width=True)

            with col_right:
                image_path = "https://raw.githubusercontent.com/Samuelchazy/Educative.io/19d3100db50749489689a5c21029c3499722b254/images/Toyota_5.jpg"
                st.image(image_path, use_container_width=True)

                image_path = "https://raw.githubusercontent.com/Samuelchazy/Educative.io/19d3100db50749489689a5c21029c3499722b254/images/Toyota_6.jpg"
                st.image(image_path, use_container_width=True)

        return content

    else:
        st.error('User Manuel not found')
        return None
