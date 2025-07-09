#streamlit run ...
import streamlit as st
import time 
from agent_lsm_streamlit import response
import pyperclip
import datetime

now = datetime.datetime.now()

st.title("I am your personal agent 🤖")
st.title("Ask me anything!")

#첫메세지
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything! 👇"}]

#응답
def response_gen(query):
    AI_reponse = response(query)

    for word in AI_reponse.split():
        yield word + " "
        time.sleep(0.05)

now = now.strftime("%H:%M") #시간 


# ---------- 과거 메시지 재렌더링 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "time" in msg:            
            st.markdown(msg["time"])

#---------prompt----------#

if prompt := st.chat_input("Ask anything 😄"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        st.markdown(now)
    

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt,"time": now}) #수정
    #st.session_state.messages.append({"role": "user", "content": time_now_US})
    
#-----------response-------#

    with st.chat_message("assistant"):
        AI_response = st.write_stream(response_gen(prompt)) 
        st.markdown(now)

 
    AI_response_txt = ''.join(AI_response)

    #session 저장
    st.session_state["last_ai"] = AI_response_txt
    st.session_state.messages.append({'role':'assistant','content':AI_response_txt,"time":now})
    #st.session_state.messages.append({'role':'assistant','content':time_now_AI})
    # if st.button('copy'):
    #     pyperclip.copy(AI_response_txt)



#copy AI_reponse
last_ai = st.session_state.get('last_ai')
if last_ai:
    if st.button("copy", icon= ":material/content_copy:"):
        pyperclip.copy(last_ai)
        st.toast("✅ 클립보드에 복사 완료!")

