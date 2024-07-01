import requests
import json

url = "http://localhost:5000//api/deployment"
data = {
    "message": "tell me about the insects that attack coconut crop?",
    "context_length":2048
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, data=json.dumps(data), headers=headers)

output  =  response.json()
result = output["response"]
print("final response: " ,result)