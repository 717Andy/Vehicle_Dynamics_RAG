import os
import logging
import warnings
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import logging as hf_logging

# 3.1 Suppress Noisy Logs
logging.basicConfig(level=logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# 3.2 ChatGPT API Credentials
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 3.3 Parameters
chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 20
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_m = 8

# Initialize Models
embedder = SentenceTransformer(model_name)
reranker = CrossEncoder(cross_encoder_name)

# 3.4 Read Document
with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 3.5 Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    separators=['', '\n', ' ', ''],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
chunks = text_splitter.split_text(text)

# 3.6 Embed & Build FAISS Index
embeddings = embedder.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

# 3.7 Retrieval Function
def retrieve_chunks(question, k=top_k):
    q_vec = embedder.encode([question])
    q_arr = np.array(q_vec).astype('float32')
    distances, indices = faiss_index.search(q_arr, k)
    return [chunks[i] for i in indices[0]]

# 3.8 Cross-Encoder Re-Ranker
def dedupe_preserve_order(items):
    seen = set()
    return [x for x in items if not (x in seen or seen.add(x))]

def rerank_chunks(question, candidate_chunks, m=top_m):
    pairs = [(question, chunk) for chunk in candidate_chunks]
    scores = reranker.predict(pairs)
    # Sort indices by score descending
    sorted_indices = np.argsort(scores)[::-1]
    top_chunks = [candidate_chunks[i] for i in sorted_indices[:m]]
    return dedupe_preserve_order(top_chunks)

# 3.9 Q&A with ChatGPT
def answer_question(question):
    candidates = retrieve_chunks(question)
    relevant_chunks = rerank_chunks(question, candidates, m=top_m)
    context = "\n\n".join(relevant_chunks)
    
    system_prompt = "You are a knowledgeable assistant that answers questions based on the provided context. If the answer is not in the context, say you don’t know."
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    
    response = client.chat.completions.create(
        model="gpt-4o", # Updated from gpt-5
        messages=[{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}],
        temperature=0.0,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# 3.10 Interactive Loop
if __name__ == "__main__":
    print("System ready. Enter 'exit' or 'quit' to end.")
    while True:
        question = input("Your question: ")
        if question.lower() in ("exit", "quit"):
            break
        print("Answer:", answer_question(question))