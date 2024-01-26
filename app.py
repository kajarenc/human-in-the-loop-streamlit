import streamlit as st
from graph import app as langgraph_app

from langchain_core.messages import HumanMessage, FunctionMessage, AIMessage

st.header("Human in the Loop demo! 🔁")

left, _, right = st.columns([80, 2, 18])

if "langgraph_state" not in st.session_state:
    st.session_state["langgraph_state"] = {
        "special_messages": [],
        "user_answer": False,
        "user_approve_requested": False,
        "user_approve_context": "",
    }

print("AAAAAA MAMA JAN, RERUN REQUSTED!")
print(st.session_state)

with left:
    if not st.session_state["langgraph_state"]["user_approve_requested"]:
        chat_container = st.container(height=625)

        prompt = st.chat_input("Say something")

        if prompt:
            print(f"PROMPT NOT EMPTY!!!! {prompt}")
            st.session_state["langgraph_state"]["special_messages"].append(
                HumanMessage(content=prompt)
            )

            inputs = {
                "special_messages": st.session_state["langgraph_state"][
                    "special_messages"
                ],
                "user_answer": False,
                "user_approve_requested": False,
                "user_approve_context": "",
            }

            print("JUST INPUTZZ")
            print(inputs)

            result = langgraph_app.invoke(inputs)
            print("RESULT")
            print(result)
            st.session_state["langgraph_state"] = result

        for message in st.session_state["langgraph_state"]["special_messages"]:
            if isinstance(message, HumanMessage):
                with chat_container.chat_message("human"):
                    st.write(message.content)
            elif isinstance(message, FunctionMessage):
                with chat_container.chat_message("assistant", avatar="🧞‍♂️"):
                    st.write(str(message)[:500])
            elif isinstance(message, AIMessage):
                with chat_container.chat_message("assistant", avatar="🤖"):
                    st.write(message.content)
            else:
                st.write("AAAAAAA")
                st.write(message)
        print("OOOOOOOOOOO")
        print(st.session_state["langgraph_state"])

with right:
    if st.session_state["langgraph_state"]["user_approve_requested"]:
        user_input = st.checkbox(
            st.session_state["langgraph_state"]["user_approve_context"]
        )
        if user_input:
            st.session_state["langgraph_state"]["user_answer"] = True
        else:
            st.session_state["langgraph_state"]["user_answer"] = False


if (
    st.session_state["langgraph_state"]["user_approve_requested"]
    and st.session_state["langgraph_state"]["user_answer"] == True
):
    real_input = st.session_state["langgraph_state"]
    print("FOOTER INPUT")
    print(real_input)
    new_result = langgraph_app.invoke(real_input)
    print("FOOTER RESULT")
    print(new_result)
    st.session_state["langgraph_state"] = new_result
    st.session_state["langgraph_state"]["user_approve_requested"] = False
    st.session_state["langgraph_state"]["user_answer"] = False
    st.session_state["langgraph_state"]["user_approve_context"] = ""
    st.rerun()
