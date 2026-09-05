from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

model = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",task="text-generation",temperature = 0.3,max_new_tokens = 100)

chat_model = ChatHuggingFace(llm=model)

result = chat_model.invoke("Write a 5 line poem on rain.")
print(result.content)