import json
import base64
from pydantic import BaseModel, Field
from ollama import chat

# Define structured response model
class Rating(BaseModel):
    rate: float = Field(description="The rating as a float number in the range 0.0 to 1.0")

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
    prompt_img = (
        "The following query and image are about an item. "
        "Estimate the relevance of the query and the image as a floating point number in a scale of 0.0 to 1.0:\n"
        "Query: {query}\n"
        "Image URL: {image_url}\n"
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
        try:
            url = f"data:image/jpg;base64,{self.image_to_base64(img_path)}"
        except FileNotFoundError:
            print("This business does not have images")
            return Rating(rate=0.0)
        user_prompt = self.prompt_img.format(
            query=query,
            image_url=url,
            image_relevance_definition=self.image_relevance_definition
        )
        return self._call_llm(user_prompt)
    
    def image_to_base64(self, img_path: str) -> str:
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")

# # Example Usage:
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

#     # Replace 'path/to/image.jpg' with an actual image file path on your system.
#     image_score = api.call_llm_image(
#         "Looking for a scenic view", 
#         "path/to/image.jpg"
#     )
#     print("Image Relevance:", image_score)
