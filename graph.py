import os
from dotenv import load_dotenv


from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
import streamlit as st

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, END

load_dotenv()
print(os.getenv("LANGCHAIN_ENDPOINT"))

tools = [TavilySearchResults(max_results=2)]
tool_executor = ToolExecutor(tools)
model = ChatOpenAI(temperature=0, streaming=True)

functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)


class AgentState(TypedDict):
    special_messages: Annotated[Sequence[BaseMessage], operator.add]
    user_answer: bool
    user_approve_requested: bool
    user_approve_context: str


def entry_point(state):
    return {}


def entry_node_transitions(state):
    special_messages = state["special_messages"]
    user_answer = state["user_answer"]
    user_approve_requested = state["user_approve_requested"]
    last_message = special_messages[-1]
    # If there is a function call, then user approved, we call action
    if (
        "function_call" in last_message.additional_kwargs
        and user_answer is True
        and user_approve_requested
    ):
        return "go_to_action"

    if isinstance(last_message, HumanMessage) and not user_approve_requested:
        return "go_to_agent"
    else:
        print("UNEXPECTED GO TO END END END!!!")
        return "end"


def call_model(state):
    special_messages = state["special_messages"]
    response = model.invoke(special_messages)

    if "function_call" in response.additional_kwargs:
        action = ToolInvocation(
            tool=response.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(
                response.additional_kwargs["function_call"]["arguments"]
            ),
        )
        user_approve_context = f"Continue with: {action}?"
        user_approve_requested = True

        # We return a list, because this will get added to the existing list
        return {
            "special_messages": [response],
            "user_approve_requested": user_approve_requested,
            "user_approve_context": user_approve_context,
        }
    return {
        "special_messages": [response],
    }


def call_tool(state):
    special_messages = state["special_messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = special_messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )

    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {
        "special_messages": [function_message],
        "user_approve_requested": False,
        "user_approve_context": "",
    }


workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("entry_point", entry_point)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("entry_point")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "entry_point",
    # Next, we pass in the function that will determine which node is called next.
    entry_node_transitions,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        "go_to_action": "action",
        "go_to_agent": "agent",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")
workflow.add_edge("agent", END)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()
