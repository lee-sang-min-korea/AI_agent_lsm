import os

from langchain.tools import tool
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_community.document_loaders import FireCrawlLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

#path
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.join(current_dir,'web_chroma_db')


#url
urls = "https://duseongchang.github.io/"

def web_vector_store():

    #API key
    api_key = os.getenv("FIRECRAWL_API_KEY") #웹에서의 내용을 벡터로 변환
    if not api_key:
        raise ValueError("No FIRECRAWL_API_KEY")    
    
    #Crawl web
    loader = FireCrawlLoader(api_key = api_key,url=urls,mode='scrape')
    
    docs = loader.load() #실제로 웹에서 데이터를 가져오는 실행 함수 

    #list -> string
    for doc in docs:
        for key, value in doc.metadata.items(): #key->author, value-> 내용
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))   


    #split doc
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100) 
    split_docs = text_splitter.split_documents(docs)

    #embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    #embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


    #chroma web db
    db = Chroma.from_documents(
        split_docs, embeddings, persist_directory=web_dir
    )

    db.persist()


@tool
def query_vector_store(input: str) :
    """
    Web Vector DB에서 쿼리에 가장 유사한 문서 3개를 검색.
    결과는 page_content와 metadata를 포함한 dict.
    """
    if not os.path.exists(web_dir):
        web_vector_store()
    
    #embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    #load db
    db = Chroma(persist_directory = web_dir, embedding_function= embeddings)
    
    retriever = db.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":3}
    )

    relevant_docs = retriever.invoke(input)

    result =[]
   
    for doc in relevant_docs:
        result.append({"content": doc.page_content, "metadata":doc.metadata})

    return result

