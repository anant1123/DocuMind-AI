from langchain_community.embeddings import HuggingFaceEmbeddings

print("Downloading and caching embedding model...")

# This line will download the model from Hugging Face and save it
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Model download complete.")