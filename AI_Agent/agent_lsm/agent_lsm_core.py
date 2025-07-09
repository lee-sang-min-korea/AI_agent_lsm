import wikipediaapi

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from conversation import ask_with_rag,chat_memory
from agent_lsm.web_crawling_test import query_vector_store
import conversation


# Load environment variables from .env file
load_dotenv()

#Tools

def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    wiki = wikipediaapi.Wikipedia(user_agent='agent')
    page = wiki.page(query)

    if not page.exists:
        return f"Can't find about '{query}"
    
    return page.summary[:400]
    
#define tools
tools =[
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
    ),

    Tool(
        name = "rag_publication",
        func = ask_with_rag ,
        description = "Useful when user want you to answer based on publication or askes about Cultural VLM Benchmarking, Rank-Insensitive Quantization Compensation, Image History Integration in MDRG ",
    ),

    Tool(
        name = "web_crawling",
        func = query_vector_store,
        description = "Useful when user askes about Du-Seong Chang"
    )

]

#prompt
prompt = hub.pull("hwchase17/structured-chat-agent")

#llm
llm = ChatOpenAI(model="gpt-4o")

#chat history
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

#수정
conversation.chat_memory = memory #glbl에 전달

#agent 선언 
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, # type: ignore
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Wikipedia,rag_publication,web_scroling."
memory.chat_memory.add_message(SystemMessage(content=initial_message))


# Chat Loop 
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # chat history
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # invoke
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])
    #print("chat history: ", memory.chat_memory.messages)

    # add to chat history
    memory.chat_memory.add_message(AIMessage(content=response["output"]))