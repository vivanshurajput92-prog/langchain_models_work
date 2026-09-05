from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"   
)

docs = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

doc_embeded = embedding.embed_documents(docs)

ques = input()
ques_embeded = embedding.embed_query(ques)

similarity = cosine_similarity([ques_embeded],doc_embeded)
score = similarity[0]

best_index = np.argmax(score)
best_match = docs[best_index]
best_score = score[best_index]

print(score)
print(best_match)
print(best_score)