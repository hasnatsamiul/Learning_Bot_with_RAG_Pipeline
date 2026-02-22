from langchain_community.embeddings import HuggingFaceEmbeddings

# Free local embedding model
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)