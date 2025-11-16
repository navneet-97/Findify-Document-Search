from sentence_transformers import SentenceTransformer
from transformers import pipeline

SentenceTransformer('all-MiniLM-L6-v2')
pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
