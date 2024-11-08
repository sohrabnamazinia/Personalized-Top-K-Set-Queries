from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import tool

class Rating(BaseModel):
    rate : float = Field("The rating as a float number in the range 0.0 to 1.0") 

class LLMApi:
    RELEVANCE = "relevance"
    DIVERSITY = "diversity"
    prompt_relevance = "The following query and document (review, text, etc) are about an item. Estimate the relevance of the query and the review as a floating point number in a scale of 0.0 to 1.0:\Query: {query}\nReview: {d}\n The definition of the relevance is fully user-defined as follows:{relevance_definition}"
    prompt_diversity = "The following two documents (review, text, etc) are about an item. Estimate the diversity of these two documents as a floating point number in a scale of 0.0 to 1.0:\nDocument 1: {d1}\nDocument 2: {d2}\n The definition of the diversity is fully user-defined as follows:{diversity_definition}"
    # NOTE: set this false if not needed
    def __init__(self, is_output_discrete=True, relevance_definition="Relevance", diversity_definition="Diversity") -> None:
        self.model_name = "gpt-4o-mini"
        #self.model_name = "gpt-4-turbo"
        self.model = ChatOpenAI(model=self.model_name).with_structured_output(Rating)
        self.prompt_relevance = ChatPromptTemplate.from_template(LLMApi.prompt_relevance)
        self.prompt_diversity = ChatPromptTemplate.from_template(LLMApi.prompt_diversity)
        self.relevance_definition = relevance_definition
        self.diversity_definition = diversity_definition
        self.is_output_discrete = is_output_discrete
        
    
    def call_llm_relevance(self, query, d):
        chain = self.prompt_relevance | self.model 
        result = chain.invoke({"query": query, "d": d, "relevance_definition":self.relevance_definition})
        if not self.is_output_discrete:
            return result.rate
        else:
            return round(result.rate, 1)
    
    
    def call_llm_diversity(self, d1, d2):
        chain = self.prompt_diversity | self.model 
        result = chain.invoke({"d1": d1, "d2": d2, "diversity_definition": self.diversity_definition})
        if not self.is_output_discrete:
            return result.rate
        else:
            return round(result.rate, 1)

# api = LLMApi(relevance_definition="it should magnify well")
# result = api.call_llm_relevance("I need a camera which can zoom very much", "this camera was good, it could magnify very well")
# print(result)
        