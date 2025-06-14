from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

print(client)
print("##############")

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

print(db)
print("######")
query = "What are some of the success stories of LeadWalnut?"

docs = db.similarity_search_with_score(query=query, k=5)
for i in docs:
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})