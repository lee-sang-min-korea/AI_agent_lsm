# from langchain_community.document_loaders import FireCrawlLoader
# import os
# from dotenv import load_dotenv

# load_dotenv()  # .env에서 API 키 로드
# api_key = os.getenv("FIRECRAWL_API_KEY")

# # 크롤링할 URL
# url = "https://www.metroseoul.co.kr/article/20221206500346"

# # 크롤 시도
# loader = FireCrawlLoader(api_key=api_key, url=url, mode='scrape')
# docs = loader.load()

# # 결과 확인
# print("크롤된 문서 수:", len(docs))
# for i, doc in enumerate(docs, 1):
#     print(f"\n📄 문서 {i}")
#     print(f"메타데이터: {doc.metadata}")
#     print(f"내용:\n{doc.page_content[:500]}")
import requests
from bs4 import BeautifulSoup

url = "https://duseongchang.github.io/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

paragraphs = soup.find_all('p')
for i, p in enumerate(paragraphs, 1):
    text = p.get_text(strip=True)
    if text:
        print(f"[{i}] {text}")
