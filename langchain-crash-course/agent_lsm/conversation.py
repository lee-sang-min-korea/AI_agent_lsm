chat_memory = None


import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from typing import List

# Load environment variables from .env
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir,"rag","chroma_db")

#embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#load vector store
db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

#********************retriver******************** 

#후보1(핵심내용만)
# retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={
#     "score_threshold": 0.5
# })
#후보2:(다양한 시각, 중복 x): max marginal relevance
retriever = db.as_retriever(search_type="mmr", search_kwargs={
    "fetch_k": 20,
    "k": 5,
    "lambda_mult": 0.5
})
# #후보3 (단순 질문-답변)
# retreiever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

#llm으로 gpt - 40 사용
llm = ChatOpenAI(model="gpt-4o")


#*******************맥락을 고려한 질문 prompt***********

#AI로 하여금 이전의 대화를 참고해서 답변을 하게 하기 위한 prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
#이전의 대화 내용 + 다음 질문 
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"), #이전의 대화 내용 삽입
        ("human","{input}") #현재 질문 
    ]
)

#이전의 대화 내용을 바탕으로 질문 해석 & 생성(논문 검색 -> 논문의 관련 부분 전달)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

#*******************질문에 대한 대답 prompt***********
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}" #<- retriever가 논문을 찾아 출력한 context  
)

#대답 prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

#qa prompt를 llm 모델에 전달 -> 최종 답변을 생성
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

#이전 대화를 고려한 retriever와 답변 생성 chain을 수행하는 chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

@tool
def ask_with_rag(input: str) -> str:
    """RAG 기반으로 질문에 대답. 입력은 질문 str."""
    global chat_memory
    result = rag_chain.invoke({"input": input,"chat_history": chat_memory.chat_memory.messages}) # type: ignore
    return result['answer']


# def continual_chat():
#     print("채팅을 시작하세요, 'exit'을 통해 대화를 끝내세요")
#     chat_histroy =[]
#     while True: #exit 전까지
#         query = input("User: ")
        
#         if query.lower() == "exit":
#             break

#         result = rag_chain.invoke({"input": query,"chat_history": chat_histroy})
        
#         for source in result["context"]:
#             print(f"Source: {source.metadata}")

#         print(f"AI: {result['answer']}")

#         chat_histroy.append(HumanMessage(content = query))
#         chat_histroy.append(SystemMessage(content = result['answer']))
        
        


