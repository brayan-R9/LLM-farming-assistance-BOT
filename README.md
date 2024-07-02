# custom-fine-tuned-RAG-LLM
custom dataset fine tuned - quantized - RAG based LLM for agriculture assistance with RESTful API deployment

# Problem Statement
To create a LLM model with data regarding information about the farming of coconuts crops.

# Solution
### Transformer
**Self attention:**
Finding the important word in the sentence.
Query, key, value
=softmax(QK/√(Dk))V

Dk - dimensionality
Q - query
K - key/response
V - value
Softmax(a) = e^a/∑e^b

**layers:**

1.Input embedding - converting to vectors ( numbers)
2.Positional encoding - maps to another number
 ``` 
    Pe(pos,2i) = sin(pos/100002i/dmodel)  
	Pe (pos,2i+1)= cos(pos/100002i/dmodel)  
	0<= i < d/2
 ```
3.Self attention - gives a new set of numbers
                   Q = xwq (seeking vector)
                   K = xwk (relevant vector)
                   V = xwv (validator) (aggregation score) features of data
                   Softmax(qk/root(d))v
4.Feed forward - in-depth feature extraction
```
    FFN = Max(0,xw1+b1)w2+ b2  
            		    ^        ^  
            		  Hidden   output  
	Bring to higher dimension
```
5.Normalization
  `Layer norm = ((x−u)/σ)γ+β `  
  Gamma and beta acquired through model training
6.Multiheaded attention
   `Multihead(q,k,v) = Concate(head1,head2,…)`  
	 `Headi=attention(QWiq,KWik,VWiv)` 
    Perform more feature extraction
7.Output

### Architecture:
1.transformer architecture: https://colab.research.google.com/drive/1JIuC9poeG3TPbL7-sb6eNnPua7pG43xM?usp=sharing
- Generating text through transformers  
- Vocab creation  
- Indexing token and vice versa  
- Transformer: embedding -> forward

2.bpe tokenization: https://colab.research.google.com/drive/1K4RlgFT-b3aEdqM9A0s9K11DGpTlt3uM?usp=sharing
- Split the sentence and check each adjacent elements
- Create pairs and their frequency count
- Merge pairs based on highest freq

*Using sentencepiece module*  
- Train using data  
- Use the model to encode as pieces  
```
original sentence: I have a dog
tokenized sentence: ['_','I','_ha','v','e','_','a','_dog']
```

### Fine tuning
**Low rank adaptation:**   
Create an adaptor model using existing layers, and merge the adaptor model with original model, Reducing the rank of large matrix to get smaller rank matrix to reduce computational complexity 

LLM fine-Tuning: https://colab.research.google.com/drive/1jFJ83m0gu2nZWNGVGn8kI-oVPRkv98pa?usp=sharing

## Limitations
- Blunder generation
- Hallucination
- Unexpected response for unknown inputs
  

## Applying RAG and overcoming limitations

Most of the limitation can be solved using RAG   
RAG helped train on newly added data  
Use langchain and db vectors for optimization

quantization with RAG: https://colab.research.google.com/drive/1RxTzleHm7a29v69MI7x0qUyqBOwfAIlz?usp=sharing

### Limitations

Could not perform well when greeting input was given. It responded with the answers for random Questions from the data.

## Further research

Can be furher improved by adding:
- Greetings
- Giving the responses a flow rather than sounding robotic
- Building an interface easier user interaction

## Deployment

### Deployment as an OpenAI Compatible API

#### Install vLLM + Haystack

- we install vLLM using pip 
- for production use cases, there are many other options, including Docker 

``` 
!pip install vllm haystack-ai 
```

```py
# we prepend "nohup" and postpend "&" to make the Colab cell run in background
! nohup python -m vllm.entrypoints.openai.api_server \
				--model '/content/drive/MyDrive/Colab Notebooks/addon GENAI/final_weights_new' \
				--dtype auto \
				--max-model-len 2048 \
				> vllm.log &
```
``` py
# we check the logs until the server has been started correctly
!while ! grep -q "Application startup complete" vllm.log; do tail -n 1 vllm.log; sleep 5; done
```
```py
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
import string
import random

# initializing size of string
N = 20

# using random.choices()
# generating random strings
res = ''.join(random.choices(string.ascii_uppercase +
							string.digits, k=N))

generator = OpenAIChatGenerator(
	api_key=Secret.from_token(res),  # for compatibility with the OpenAI API, a placeholder api_key is needed
	model="/content/drive/MyDrive/Colab Notebooks/addon GENAI/final_weights_new",
	api_base_url="http://localhost:8000/v1",
	generation_kwargs = {"max_tokens": 1024}
)

```
```py
messages = []

while True:
msg = input("Enter your message or Q to exit\n🧑 ")
if msg=="Q":
	break
messages.append(ChatMessage.from_user(msg))
response = generator.run(messages=messages)
assistant_resp = response['replies'][0]
print("🤖 "+assistant_resp.content)
messages.append(assistant_resp)
```
### GGUF Model Deployment Guide

Sample deployment configuration

#### Prerequisites

- Python 3.8 or higher

#### Installation
1. Install the required Python packages.

#### Ubuntu CPU

```sh
pip install llama-cpp-python
pip install flask
```
#### Ubuntu with CUDA

```sh
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
pip install flask
```

#### Windows

1.Download and install Anaconda python from [here](https://www.anaconda.com/download)

```sh
conda create -n deeplearning python=3.8
conda activate deeplearning
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install llama-cpp-python
pip install flask
```

2.Download the model required from [here](https://drive.google.com/file/d/126tBo3ulCjkiOMJ9UpBBzFp18fNSOPid/view?usp=sharing).

*the model file could'nt be uploaded to github due to size limits, rest of the files are uploaded in this repository*

*the data for the model is present in coconut.xlsx*

#### CPU 
1. Run the `app_cpu.py` script to start the Flask server.

```sh
python app_cpu.py
```
#### CUDA GPU
1. Run the `app_cuda_gpu.py` script to start the Flask server.

```sh
python app_cuda_gpu.py
```

2. In a new terminal, run the `post_request.py` script to send a POST request to the server.

```sh
python post_request.py
```

#### Code Explanation

`app.py` is the main server file that uses Flask to create a web API. It uses the Llama library to generate responses based on the input message.

`post_request.py` is a script that sends a POST request to the server with a message. The server then uses the Llama library to generate a response and sends it back to the client.

#### Sample Request

You can send a POST request to `http://localhost:5000/api/deployment` with the following JSON body:

```json
{
    "message": "tell me about the insects that attack coconut crops?"
}
```

The server will respond with the AI-generated response.

