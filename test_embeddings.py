from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

text = "what's the treatment for hypertension?"

vec = model.encode(text)

print("Vector length:", len(vec))
print("First 5 numbers:", vec[:5])