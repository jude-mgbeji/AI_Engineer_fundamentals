import streamlit as st
from openai import OpenAI
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(page_title="LLM Engineering Prototype Development", page_icon=":robot_face:")
st.title("Interview Chatbot")

if 'user_detail_setup_complete' not in st.session_state:
    st.session_state.user_detail_setup_complete = False
if 'user_message_count' not in st.session_state:
    st.session_state.user_message_count = 0
if 'is_feedback_shown' not in st.session_state:
    st.session_state.is_feedback_shown = False
if 'is_interview_complete' not in st.session_state:
    st.session_state.is_interview_complete = False

def user_detail_setup_complete():
    st.session_state.user_detail_setup_complete = True

def show_feedback():
    st.session_state.is_feedback_shown = True

if not st.session_state.user_detail_setup_complete:
    st.subheader('Personal Information', divider='rainbow')

    if 'name' not in st.session_state:
        st.session_state["name"] = ""
    if 'experience' not in st.session_state:
        st.session_state['experience'] = ""
    if 'skills' not in st.session_state:
        st.session_state['skills'] = ""
    if 'level' not in st.session_state:
        st.session_state['level'] = "Intern"
    if 'position' not in st.session_state:
        st.session_state.position = "AI Engineer"
    if 'company' not in st.session_state:
        st.session_state['company'] = "Google"

    st.session_state["name"]  = st.text_input("Name", value= st.session_state["name"] , max_chars= 50, placeholder="Enter your name")
    st.session_state['experience']  = st.text_area("Experience", value=st.session_state['experience'] , max_chars= 200, placeholder="Describe your experience")
    st.session_state['skills'] = st.text_area("Skills", value=st.session_state['skills'], max_chars= 200, placeholder="List your skills")

    st.subheader('Company and Position', divider='rainbow')
    col1, col2 = st.columns(2)
    with col1:
        st.session_state['level'] = st.radio("Choose level",key='visibility', options=["Intern", "Junior", "Mid", "Senior"], index=0)
    with col2:
        st.session_state['position'] = st.selectbox("Choose Position", options=["AI Engineer", "Data Scientist", "Data Analyst", 'Data Engineer'], index=0)

    st.session_state['company'] = st.text_input("Company", value= st.session_state['company'], max_chars= 50, placeholder="Enter company name")

    if st.button("Start Interview", on_click=user_detail_setup_complete, type="primary"):
        st.write("Setup complete. Starting interview...")

if st.session_state.user_detail_setup_complete and not st.session_state.is_interview_complete and not st.session_state.is_feedback_shown: 
    st.subheader('Interview', divider='rainbow')
    st.info(f'''Hello {st.session_state["name"]}, Start by introducing yourself...''', icon="ℹ️")

    PROMPT_TEMPLATE = f'''
    You are a Lead {st.session_state['position']} at {st.session_state['company']} that interviews an applicant called {st.session_state['name']} 
    with experience {st.session_state['experience']} and skills {st.session_state['skills']}.
    You should Interview them for the position {st.session_state['level']} {st.session_state['position']} at company {st.session_state['company']}.
    '''

    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o"

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": PROMPT_TEMPLATE}
        ]

    for message in st.session_state.messages:
        if message['role'] != "system":
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    if st.session_state.user_message_count <= 5:
        if prompt := st.chat_input("Your answer..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.user_message_count < 4:
                with st.chat_message("assistant"):
                    response_stream = openai_client.chat.completions.create(
                        model=st.session_state.openai_model,
                        messages=st.session_state.messages,
                        stream=True
                    )
                    respponse = st.write_stream(response_stream)
            
                st.session_state.messages.append({"role": "assistant", "content": respponse})
            
            st.session_state.user_message_count += 1

    if st.session_state.user_message_count >= 5:
        st.session_state.is_interview_complete = True
        st.success("Interview complete.", icon="✅")

if st.session_state.is_interview_complete and not st.session_state.is_feedback_shown:
    if st.button("Show Feedback", on_click=show_feedback, type="primary"):
        st.write("Generating Feedback.")

if st.session_state.is_feedback_shown:
    st.subheader('Feedback', divider='rainbow')
    st.info("Feedback will be shown here.", icon="ℹ️")

    conversation_history ='\n'.join(f"{message['role']}: {message['content']}" for message in st.session_state.messages)

    feedback_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    feedback_prompt = '''You are a helpful tool that provides feedback on an interview performance.
    Before the feedback, give a score from 1 to 10.
    Follow this format:
    Overall Score: //Your score
    Feedback: //Your feedback
    Give only the feedback, do not ask any additional questions.
     '''
    
    feedback_response = feedback_client.chat.completions.create(
        model=st.session_state.openai_model,
        messages=[
            {"role": "system", "content": feedback_prompt},
            {"role": "user", "content": f'This is the interview you need to evaluate. Keep in mind that you are only a tool that provides feedback. \n\n{conversation_history}'}
        ]
    )

    feedback = feedback_response.choices[0].message.content
    st.write(feedback)

    if st.button("Restart Interview", on_click=lambda: st.session_state.clear(), type="primary"):
        streamlit_js_eval(js_expression="parent.window.location.reload()")


