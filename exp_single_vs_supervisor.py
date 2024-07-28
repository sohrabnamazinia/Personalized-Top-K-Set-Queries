from langchain_openai import ChatOpenAI
from read_data import read_data
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

# set input (each prompt = 1 task)
sequence = read_data("input.txt")
model_name = "gpt-3.5-turbo"
prompt_1 = "extract the top 3 features from the following reviews about a prdocut. \n {reviews}"
prompt_2 = "Evaluate the quality of the product based on the following reviews: (only give one single number from 1-10) \n {reviews}"
prompts = [prompt_1, prompt_2]

def call_single_llm(sequence, model_name, prompts):
    model = ChatOpenAI(model=model_name)
    outputParser = StrOutputParser()
    for prompt in prompts:
        prompt = ChatPromptTemplate.from_template(prompt)
        chain = prompt | model | outputParser
        result = chain.invoke({"reviews" : sequence})
        print(result)

@tool
def extract_features(query: str) -> str:
    """extract the top 2 features from the input sequence of reviews\n"""
    model = ChatOpenAI(model=model_name)
    prompt = ChatPromptTemplate.from_template("extract the top-3 features of the product that has been described by the following reviews: {reviews}")
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    result = chain.invoke({"reviews": query})
    return result

@tool
def estimate_quality(query: str) -> str:
    """estimate the numerical quality of a product from 1-10 based on the input sequence of reviews\n"""
    model = ChatOpenAI(model=model_name)
    prompt = ChatPromptTemplate.from_template("only return a numerical value from 1-10 that estimates the quality of this product based on the following reviews: {reviews}")
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    result = chain.invoke({"reviews": query})
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
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def route(options):
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    return function_def
        
def build_graph(feature_node, quality_node, supervisor_chain, members):
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("feature_extractor", feature_node)
    graph_builder.add_node("quality_estimator", quality_node)
    graph_builder.add_node("supervisor", supervisor_chain)

    for member in members:
        graph_builder.add_edge(member, "supervisor")

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    graph_builder.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

    graph_builder.set_entry_point("supervisor")
    graph = graph_builder.compile()
    return graph

def call_supervision_llm(sequence, model_name, prompts):
    model = ChatOpenAI(model=model_name)
    members = ["feature_extractor", "quality_estimator"]
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH.")
    
    options = ["FINISH"] + members
    function_def = route(options=options)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    supervisor_chain = (
        prompt
        | model.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser())
    
    feature_agent = create_agent(model, [extract_features], "You are only a top-3 feature extractor.")
    feature_node = functools.partial(agent_node, agent=feature_agent, name="feature-extractor")

    quality_agent = create_agent(
        model,
        [estimate_quality],
        "You evaluate the quality of a product and only return one single number from 1-10 and don't print anything else",
    )
    quality_node = functools.partial(agent_node, agent=quality_agent, name="quality-estimator")

    graph = build_graph(feature_node, quality_node, supervisor_chain, members)
    messages = []
    for prompt in prompts:
        prompt = ChatPromptTemplate.from_template(prompt)
        messages.append(prompt.invoke({"reviews" : str(sequence)}).to_messages()[0])
    for s in graph.stream({"messages": messages}):
        if "__end__" not in s:
            print(s)
            print("----")

single_llm_start_time = time.time()
call_single_llm(sequence, model_name, prompts)
single_llm_time = time.time() - single_llm_start_time

supervision_llm_start_time = time.time()
call_supervision_llm(sequence, model_name, prompts)
supervision_llm_time = time.time() - supervision_llm_start_time

print("Time taken for single LLM: ", single_llm_time)
print("Time taken for supervisor LLM: ", supervision_llm_time)