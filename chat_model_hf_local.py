from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = 'text-generation',
    pipeline_kwargs={
        "temperature":0.1,
        "max_new_tokens": 256,
        "clean_up_tokenization_spaces": False
    }
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("How can I fine-tune a open source llm?")
print(result.content)