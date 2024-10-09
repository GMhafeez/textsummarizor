from transformers import  pipeline
import os
import torch
from dotenv import load_dotenv
from torch.nn import functional as F
load_dotenv()
model_name="meta-llama/Llama-3.2-1B"
access_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
access_token_HF_TOKEN = os.getenv("HF_TOKEN")
summarizer = pipeline("text-generation",model=model_name,torch_dtype=torch.bfloat16)
result = summarizer("hello my name is ghulam Mustafa")

print(result)