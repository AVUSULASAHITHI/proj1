import gradio as gr
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM


Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


Settings.llm = HuggingFaceLLM(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)


Settings.text_splitter = SentenceSplitter(
    chunk_size=150,
    chunk_overlap=50,
)

documents = SimpleDirectoryReader(
    input_files=[r"C:\Users\dell\Downloads\Girlchildedu.pdf"]
).load_data()


client = chromadb.PersistentClient(path="./chromadb")

collection = client.get_or_create_collection("pdf_docs")

vector_store = ChromaVectorStore(chroma_collection=collection)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

query_engine = index.as_query_engine(similarity_top_k=3)


def chat(query):
    response = query_engine.query(query)
    return str(response)

demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Enter your question"),
    outputs=gr.Textbox(label="Response"),
    title="RAG PDF Chatbot",
)

demo.launch()