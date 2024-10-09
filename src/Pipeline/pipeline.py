from transformers import  pipeline
import os
import torch
from torch.nn import functional as F
model_name="meta-llama/Llama-3.2-1B"
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
summarizer = pipeline("text-generation",model=model_name,torch_dtype=torch.bfloat16)
result = summarizer("hello my name is ghulam Mustafa")

print(result)