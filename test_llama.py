import json
from llamaapi import LlamaAPI

# Initialize the SDK
llama = LlamaAPI("LA-42d93c09afb24f5f896e8dd48d42493ebc1a9e1d9881441385be97b44c54dbaf")

# Build the API request
api_request_json = {
    "model": "llama3.1-70b",
    "messages": [
        {"role": "user", "content": "What is the weather like in Boston?"},
    ],
    "functions": [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "days": {
                        "type": "number",
                        "description": "for how many days ahead you wants the forecast",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
            },
            "required": ["location", "days"],
        }
    ],
    "stream": False,
    "function_call": "get_current_weather",
}

# Execute the Request
response = llama.run(api_request_json)
print(json.dumps(response.json(), indent=2))