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
sequence = read_data("input.txt")[0]
model_name = "gpt-4-turbo"
task_1 = "name the top 3 features from the following reviews about a prdocut.\n {reviews}"
task_2 = "only select one of [\"HIGH\", \"MEDIUM\", \"LOW\"] to estimate the quality of the product described by the following reviews: \n {reviews}"
system_prompt_1 = "You only name the top 3 features of the product based on the reviews"
system_prompt_2 = "You only choose one of [\"HIGH\", \"MEDIUM\", \"LOW\"] to estimate the quality of the product based on the reviews"
task_combined = "name the top 3 features from the following reviews about a prdocut. Then, use those features to choose one of [\"HIGH\", \"MEDIUM\", \"LOW\"] to estimate the quality of the product. Here are the reviews: \n {reviews}"
tasks = [task_1, task_2]
#tasks = [task_combined]

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
    """Name the top-3 features"""
    model = ChatOpenAI(model=model_name, temperature=0.1)
    prompt = ChatPromptTemplate.from_template(task_1)
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    result = chain.invoke({"reviews": query})
    return result

@tool
def estimate_quality(query: str) -> str:
    """Select one option from [HIGH, MEDUMN, LOW] as the product quality"""
    model = ChatOpenAI(model=model_name)
    prompt = ChatPromptTemplate.from_template(task_2)
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
    state["messages"].append(HumanMessage(content="You might know the answer without running any code, but you should still use your tool to get the answer."))
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
    
    feature_agent = create_agent(model, [extract_features], system_prompt_1)
    feature_node = functools.partial(agent_node, agent=feature_agent, name=members[0])

    quality_agent = create_agent(
        model,
        [estimate_quality],
        system_prompt_2,
    )
    quality_node = functools.partial(agent_node, agent=quality_agent, name=members[1])

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
call_single_llm(sequence, model_name, tasks)
single_llm_time = time.time() - single_llm_start_time

supervision_llm_start_time = time.time()
call_supervision_llm(sequence, model_name, tasks)
supervision_llm_time = time.time() - supervision_llm_start_time

print("Time taken for single LLM: ", single_llm_time)
print("Time taken for supervisor LLM: ", supervision_llm_time)