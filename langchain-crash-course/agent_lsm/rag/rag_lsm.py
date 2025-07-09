import os
import fitz

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document



#path 
current_dir = os.path.dirname(os.path.abspath(__file__))
public_dir = os.path.join(current_dir, "publication")
db_dir = os.path.join(current_dir, "chroma_db")

#embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#pdf 2 txt

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text() # type: ignore
    return text


    #pdf 파일 순회 -> 수집
publication = []
for filename in os.listdir(public_dir):
    if filename.lower().endswith(".pdf"):
        full_path = os.path.join(public_dir, filename)
        text = extract_text_from_pdf(full_path)
        publication.append(Document(page_content=text,metadata = {"source": filename}))


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(publication)

db = Chroma.from_documents(split_docs, embeddings, persist_directory = db_dir)
db.persist()