import os

# from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
# from langchain_community.document_loaders import FireCrawlLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

load_dotenv()

#path
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.join(current_dir,'web_chroma_db')


#url
urls = "https://ai.sogang.ac.kr/ai/ai06_1.html"

def web_vector_store():

    #API key
    # api_key = os.getenv("FIRECRAWL_API_KEY")
    # if not api_key:
    #     raise ValueError("No FIRECRAWL_API_KEY")    
    
    #Crawl web
    # loader = FireCrawlLoader(api_key = api_key,url=urls,mode='scrape')
    response = requests.get(urls)
    soup = BeautifulSoup(response.text, 'html.parser')

    paragraphs = soup.find_all('p')
    
    # docs = paragraphs.load()

    # #list -> string
    # for doc in docs:
    #     for key, value in doc.metadata.items(): #key->author, value-> 내용
    #         if isinstance(value, list):
    #             doc.metadata[key] = ", ".join(map(str, value))   

    #doc으로 래핑
    docs = []
    for p in paragraphs:
        text = p.get_text(strip=True)
        if text:
            docs.append(Document(page_content=text))

    print(f"doc: {docs}")
    #split doc
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100) 
    split_docs = text_splitter.split_documents(paragraphs)

    #embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    #chroma web db
    db = Chroma.from_documents(
        split_docs, embeddings, persist_directory=web_dir
    )

    # #*******db 확인*****
    # print("Loaded docs:", len(docs))
    # for doc in docs:
    #     print(doc.metadata, doc.page_content[:200])
    # #*******************
    db.persist()


web_vector_store()





#embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#load db
db = Chroma(persist_directory = web_dir, embedding_function= embeddings)


def query_vector_store(query):
    
    retriever = db.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":3}
    )

    relevant_docs = retriever.invoke(query)

   
    for i, doc in enumerate(relevant_docs,1):
        print(f"document: {i}: {doc.page_content}")
        if doc.metadata:
            print("----source----")
            print(f"source: {doc.metadata.get('source','unkown')}\n") 

query = "Who is Du-Seong Chang?"

query_vector_store(query)


# docs = db.similarity_search("", k=1000)  # 모든 문서 검색 (빈 쿼리 사용)

# for i, doc in enumerate(docs, 1):
#     print(f"[{i}] {doc.page_content[:200]}...")  # 내용 일부 미리보기
#     print(f"    ↳ metadata: {doc.metadata}")



