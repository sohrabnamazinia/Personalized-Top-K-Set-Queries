from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import tool

class Rating(BaseModel):
    rate : float = Field("The rating as a float number in the range 0.0 to 1.0") 

class LLMApi:
    RELEVANCE = "relevance"
    DIVERSITY = "diversity"
    prompt_relevance = "The following query and review are about an item. Estimate the relevance of the query and the review as a floating point number in a scale of 0.0 to 1.0:\Query: {query}\nReview: {d}"
    prompt_diversity = "The following two comments are about an item. Estimate the diversity of these two comments as a floating point number in a scale of 0.0 to 1.0:\nComment 1: {d1}\nComment 2: {d2}"
    def __init__(self) -> None:
        self.model_name = "gpt-4-turbo"
        self.model = ChatOpenAI(model=self.model_name).with_structured_output(Rating)
        self.prompt_relevance = ChatPromptTemplate.from_template(LLMApi.prompt_relevance)
        self.prompt_diversity = ChatPromptTemplate.from_template(LLMApi.prompt_diversity)
        
    @tool
    def call_llm_relevance(self, query, d):
        """Estimate the relevance of the input to the query as a float number in the scale of 0.0 to 1.0"""
        chain = self.prompt_relevance | self.model 
        result = chain.invoke({"query": query, "d": d})
        return result
    
    @tool
    def call_llm_diversity(self, d1, d2):
        """Estimate how diverse the two inputs are as a float number in the scale of 0.0 to 1.0"""
        chain = self.prompt_diversity | self.model 
        result = chain.invoke({"d1": d1, "d2": d2})
        return result
        