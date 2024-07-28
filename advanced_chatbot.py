from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=5)
tools = [tool]
llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    if len(state["messages"]) == 3:
        state["messages"][2].content[0]["type"] = "text"
        state["messages"][2].content[1]["type"] = "text"
        state["messages"][2].content[2]["type"] = "text"
        state["messages"][2].content[3]["type"] = "text"
        state["messages"][2].content[4]["type"] = "text"

    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["q", "exit", "quit"]:
        print("Bye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                print("AI: ", value["messages"][-1].content)
