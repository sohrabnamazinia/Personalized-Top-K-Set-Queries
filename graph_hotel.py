from langchain_openai import ChatOpenAI
from read_data_hotel_reviews import read_data, read_data_fake
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import HumanMessage, BaseMessage
import operator
from typing import Annotated, Sequence, TypedDict
import functools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.tools import tool
import time
from langchain_core.pydantic_v1 import BaseModel, Field

model_name = "gpt-4-turbo"
hotel_name_1 = "Azul Beach Hotel By Karisma Gourmet Inclusive"
#hotel_name = "XXX"
hotel_names = [hotel_name_1]
features = ["distance_from_attractions", "cleanness", "final_score"]
task = "Rate the hotel that has been described by the following reviews based on \"{features}\": {reviews}"
prompt_distance_from_attraction = "Rate the hotel that has been described by the following reviews only based on \"distance from attraction\": {reviews}"
prompt_cleanness = "Rate the hotel that has been described by the following reviews only based on \"cleanness\": {reviews}"
system_prompt_distance_from_attraction = "You only return a value in range 0-10 to rate the hotel only based on its distance from attraction(s)"
system_prompt_cleanness = "You only return a value in range 0-10 to rate the hotel only based on its cleanness"
system_prompt_compute_final_score = "You only compute final score of the hotel based on the individual scores already computed"
prompts = [task]

class Rating(BaseModel):
    rate : int = Field("The rating in the range 0-10") 
    
# @tool
# def compute_final_score(scores: list):
#     """Compute the final score for a hotel"""
#     avg_score = 0
#     for score in scores:
#         avg_score += int(score)
#     avg_score /= (len(features) - 1)
#     return avg_score
@tool
def compute_final_score(scores: list):
    """Compute the final score for a hotel"""
    model = ChatOpenAI(model=model_name)
    prompt = ChatPromptTemplate.from_template("Average all the individual scores and normalize it to the range [0-1] to get the final score of the hotel: {scores}")
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    result = chain.invoke({"scores": scores})
    return result

@tool
def rate_distance_from_attraction(reviews: list):
    """Rate the hotel in a scale of 0-10 only based on its distance from attraction(s)"""
    model = ChatOpenAI(model=model_name).with_structured_output(Rating)
    prompt = ChatPromptTemplate.from_template(prompt_distance_from_attraction)
    
    chain = prompt | model 

    result = chain.invoke({"reviews": str(reviews)})
    return result

@tool
def rate_cleanness(reviews: list):
    """Rate the hotel in a scale of 0-10 only based on its cleanness"""
    model = ChatOpenAI(model=model_name).with_structured_output(Rating)
    prompt = ChatPromptTemplate.from_template(prompt_cleanness)

    chain = prompt | model 

    result = chain.invoke({"reviews": str(reviews)})
    return result

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    state["messages"].append(HumanMessage(content="You might know the answer without calling any tool, but you should only use your tool to get the answer."))
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def build_nodes():
    model = ChatOpenAI(model=model_name)
    nodes = []

    distance_from_attraction_agent = create_agent(model, [rate_distance_from_attraction], system_prompt_distance_from_attraction)
    distance_from_attraction_node = functools.partial(agent_node, agent=distance_from_attraction_agent, name=features[0])
    nodes.append(distance_from_attraction_node)

    cleanness_agent = create_agent(model, [rate_cleanness], system_prompt_cleanness)
    cleanness_node = functools.partial(agent_node, agent=cleanness_agent, name=features[1])
    nodes.append(cleanness_node)

    compute_final_score_agent = create_agent(model, [compute_final_score], system_prompt_compute_final_score)
    compute_final_score_node = functools.partial(agent_node, agent=compute_final_score_agent, name=features[-1])
    nodes.append(compute_final_score_node)

    return nodes

def build_graph(nodes):
    graph_builder = StateGraph(AgentState)
    for i in range(len(features)):
        graph_builder.add_node(features[i], nodes[i])
    
    for i in range(len(features) - 1):
        graph_builder.add_edge(features[i], features[i + 1])
    
    graph_builder.set_entry_point(features[0])
    graph = graph_builder.compile()
    return graph

def call_model(reviews_dict, hotel_names, n_reviews=1):
    for hotel_name in hotel_names:
        nodes = build_nodes()
        graph = build_graph(nodes)
        reviews = reviews_dict[hotel_name][:n_reviews]
        print("*****************************************")
        print("HOTEL NAME: ", hotel_name)
        print("REVIEWS: ", str(reviews))
        messages = []
        for prompt in prompts:
            prompt = ChatPromptTemplate.from_template(prompt)
            messages.append(prompt.invoke({"reviews" : str(reviews), "features" : str(features[:-1])}).to_messages()[0])
        for s in graph.stream({"messages": messages}):
            if "__end__" not in s:
                print(s)
                print("----")


n_reviews = 1
n_hotels = 4
reviews_dict = read_data()
#reviews_dict = read_data_fake()
hotel_names = list(reviews_dict.keys())[:n_hotels]
call_model(reviews_dict, hotel_names, n_reviews)