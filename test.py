from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(model="gpt-4")
model.bind_functions()
parser = StrOutputParser()

system_template = "Based on the user description of an {category}, predict in one word what specific kind of {category} it is?"
messages = [("system", system_template), ("user", "{input}")]
prompt_template = ChatPromptTemplate.from_messages(messages)

chain = prompt_template | model | parser
result = chain.invoke({"category": "car brand", "input": "it is high-tech"})
print(result)



