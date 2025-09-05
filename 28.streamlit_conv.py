from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from streamlit_chat import message
import warnings
warnings.simplefilter("ignore")


if 'conversation' not in st.session_state:
    gpt = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    summary_memory = ConversationSummaryMemory(llm=gpt)
    st.session_state['conversation'] = ConversationChain(llm=gpt, memory=summary_memory)

conversation = st.session_state['conversation']

container = st.container()

with st.form('chat', clear_on_submit=True):
    user_prompt = st.text_input('프롬프트 입력:')
    submit_button = st.form_submit_button(label='전송')

    if submit_button and user_prompt:
        with st.spinner('GPT가 답변을 준비하는 중...'):
            result = conversation.invoke(user_prompt)
            for msg in conversation.memory.chat_memory.messages:
                role, right = ('fun-emoji', True) if msg.type == 'human' else ('bottts', False)
                with container:
                    message(msg.content, avatar_style=role, is_user=right)
