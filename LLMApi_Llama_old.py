import json
from pydantic import BaseModel, Field
from ollama import chat

class Rating(BaseModel):
    rate: float = Field(description="The rating as a float number in the range of 0.0 to 1.0")

class LLMApiLlama:
    RELEVANCE = "relevance"
    DIVERSITY = "diversity"
    prompt_relevance = (
        "The following query and document (review, text, etc) are about an item. "
        "Estimate the relevance of the query and the review as a floating point number in a scale of 0.0 to 1.0:\n"
        "Query: {query}\n"
        "Review: {d}\n"
        "The definition of the relevance is fully user-defined as follows:{relevance_definition}"
    )
    prompt_diversity = (
        "The following two documents (review, text, etc) are about an item. "
        "Estimate the diversity of these two documents as a floating point number in a scale of 0.0 to 1.0:\n"
        "Document 1: {d1}\n"
        "Document 2: {d2}\n"
        "The definition of the diversity is fully user-defined as follows:{diversity_definition}"
    )
    prompt_relevance_image = (
        "The following query and image are about an item. "
        "Estimate the relevance of the query and the image as a floating point number in a scale of 0.0 to 1.0:\n"
        "Query: {query}\n"
        "The definition of the relevance is fully user-defined as follows:{image_relevance_definition}"
    )
    
    def __init__(self, model_name="llama3.2", relevance_definition="Relevance", 
                 diversity_definition="Diversity", image_relevance_definition="Relevance"):
        self.model_name = model_name
        self.relevance_definition = relevance_definition
        self.diversity_definition = diversity_definition
        self.image_relevance_definition = image_relevance_definition

    def _call_llm(self, user_prompt: str) -> Rating:
        response = chat(
            messages=[{"role": "user", "content": user_prompt}],
            model=self.model_name,
            format=Rating.model_json_schema()
        )
        return Rating.model_validate_json(response.message.content)

    def call_llm_relevance(self, query: str, d: str) -> Rating:
        user_prompt = self.prompt_relevance.format(
            query=query, d=d, relevance_definition=self.relevance_definition
        )
        return self._call_llm(user_prompt)

    def call_llm_diversity(self, d1: str, d2: str) -> Rating:
        user_prompt = self.prompt_diversity.format(
            d1=d1, d2=d2, diversity_definition=self.diversity_definition
        )
        return self._call_llm(user_prompt)
        
    def call_llm_image(self, query: str, img_path: str) -> Rating:
        # Check if image file exists
        try:
            with open(img_path, "rb") as f:
                pass
        except FileNotFoundError:
            print("This business does not have images")
            return Rating(rate=0.0)
            
        user_prompt = self.prompt_relevance_image.format(
            query=query,
            image_relevance_definition=self.image_relevance_definition
        )
        response = chat(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [img_path]
                }
            ],
            format=Rating.model_json_schema()
        )
        return Rating.model_validate_json(response.message.content)

# Example Usage:
# if __name__ == "__main__":
#     api = LLMApiLlama()

#     relevance_score = api.call_llm_relevance(
#         "Find the best laptop", 
#         "This laptop has great performance and battery life."
#     )
#     print("Relevance:", relevance_score)

#     diversity_score = api.call_llm_diversity(
#         "Doc about traveling to Japan", 
#         "Doc about traveling to china"
#     )
#     print("Diversity:", diversity_score)

#     image_score = api.call_llm_image(
#         "cool cat", 
#         "gratisography-cool-cat-800x525.jpg"
#     )
#     print("Image Relevance:", image_score)
