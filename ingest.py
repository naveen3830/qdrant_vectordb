from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader= PyPDFLoader("data\splashtop_markdown_content_20250613_104612.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
texts = text_splitter.split_documents(documents)

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db"
)

print("Vector DB Successfully Created!")