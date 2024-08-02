from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# class Counter(BaseModel):
#     rate : int = Field("The rating in the range 0-1") 

# words = ["AB", "CD", "EF"]
# model = ChatOpenAI(model="gpt-4")
# model = model.with_structured_output(Counter)

# prompt = ChatPromptTemplate.from_template("How many words you can see totally in here: {words}")
# output_parser = StrOutputParser()

# chain = prompt | model

# result = chain.invoke({"words": str(words)})

# print(result)



    


