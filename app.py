# import databutton as db
import streamlit as st
import sys
import re
import time
# import altair

from io import BytesIO
from typing import Any, Dict, List
import pickle

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader

from typing import List, Tuple  

def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    # Initialize the PDF reader for the provided file.
    pdf = PdfReader(file)
    output = []
    
    # Loop through all the pages in the PDF.
    for page in pdf.pages:
        # Extract the text from the page.
        text = page.extract_text()
        
        # Replace word splits that are split by hyphens at the end of a line.
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        
        # Replace single newlines with spaces, but not those flanked by spaces.
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        
        # Consolidate multiple newlines to two newlines.
        text = re.sub(r"\n\s*\n", "\n\n", text)
        
        # Append the cleaned text to the output list.
        output.append(text)
    
    # Return the list of cleaned texts and the filename.
    return output, filename

def text_to_docs(text: List[str], filename: str) -> List[Document]:
    # Ensure the input text is a list. If it's a string, convert it to a list.
    if isinstance(text, str):
        text = [text]
    
    # Convert each text (from a page) to a Document object.
    page_docs = [Document(page_content=page) for page in text]
    
    # Assign a page number to the metadata of each document.
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    
    # Split each page's text into smaller chunks and store them as separate documents.
    for doc in page_docs:
        # Initialize the text splitter with specific chunk sizes and delimiters.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        
        # Split the document's text into chunks.
        chunks = text_splitter.split_text(doc.page_content)
        
        # Convert each chunk into a new document, storing its chunk number, page number, and source file name in its metadata.
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc.metadata["filename"] = filename
            doc_chunks.append(doc)
    
    # Return the list of chunked documents.
    return doc_chunks

def docs_to_index(docs, openai_api_key):
    index = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return index


def get_index_for_pdf(pdf_files, pdf_names, openai_api_key):
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents = documents + text_to_docs(text, filename)
    index = docs_to_index(documents, openai_api_key)
    return index


# # Import necessary libraries
# import databutton as db
import streamlit as st
import openai
# from my_pdf_lib import get_index_for_pdf
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from openai import ChatCompletion, APIError

import os

# Set the title for the Streamlit app
st.title("bummock RAG Chatbot")

# Set up the OpenAI API key 
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

openai.api_key = 'your-openai-api-key'

# Cached function to create a vectordb for the provided PDF files
# @st.experimental_memo
@st.cache_data
def create_vectordb(files, filenames):
    # Show a spinner while creating the vectordb
    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, openai.api_key
        )
    return vectordb

# Upload PDF files using Streamlit's file uploader
pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)
# If PDF files are uploaded, create the vectordb and store it in the session state
if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)

# Define the template for the chatbot prompt
prompt_template = """
    You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

    Keep your answer short and to the point.
    
    The evidence are the context of the pdf extract with metadata. 
    
    Carefully focus on the metadata specially 'filename' and 'page' whenever answering.
    
    Make sure to add filename and page number at the end of sentence you are citing to.
        
    Reply "Not applicable" if text is irrelevant.
     
    The PDF content is:
    {pdf_extract}
"""

# Get the current prompt from the session state or set a default value
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

# Display previous chat messages
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything")

# Handle the user's question
if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()
    
    # Search the vectordb for similar content to the user's question
    search_results = vectordb.similarity_search(question, k=7)

    # search_results
    pdf_extract = "/n ".join([result.page_content for result in search_results])

    # Update the prompt with the pdf extract
    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }

    # Add the user's question to the prompt and display it
    prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Display an empty assistant message while waiting for the response
    with st.chat_message("assistant"):
        botmsg = st.empty()

    # Call ChatGPT with streaming and display the response as it comes
    response = []
    result = ""
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=prompt, stream=True
    ):
        text = chunk.choices[0].get("delta", {}).get("content")
        if text is not None:
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)

    # Add the assistant's response to the prompt
    prompt.append({"role": "assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt
    prompt.append({"role": "assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt

