import os
from dotenv import load_dotenv
import asyncio

# LLM imports
from llama_index.llms.ollama import Ollama

# Embedding and vector store imports
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex

# Embedding pipeline imports
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader


#####
# Set up environment
#####
'''
You need to either create a .env file with the values here (OLLAMA_BASE URL and DATA_DIR)
Or just set the variables manually here if you want.
'''
load_dotenv()
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
DATA_DIR = os.getenv('DATA_DIR')


#####
# Set up variables
#####

EMBEDDING_MODEL = 'nomic-embed-text:latest'
LLM_MODEL = 'llama3.1:latest'

#####
# Set up LLM
#####
llm = Ollama(
    model=LLM_MODEL,
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
    base_url=OLLAMA_BASE_URL
)

#####
# Set up RAG data
#####
reader = SimpleDirectoryReader(input_dir=DATA_DIR)
documents = reader.load_data()
print(len(documents))
#####
# Set up ollama embedding
#####
ollama_embedding = OllamaEmbedding(
    model_name=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
    ollama_additional_kwargs={"mirostat": 0},
)

# create the pipeline with transformations
pipeline_no_vector_store = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        #HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ollama_embedding,
    ]
)


db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=100, chunk_overlap=0),
        #HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ollama_embedding,
    ],
    vector_store=vector_store,
)

### Test nodes
async def create_nodes(documents):
    #nodes = await pipeline.arun(documents=[Document.example()])
    nodes = await pipeline.arun(show_progress=True, documents=documents)
    #print(nodes)
    return nodes

#nodes = asyncio.run(create_nodes(documents=documents))

### Test nodes
def create_nodes(documents):
    #nodes = await pipeline.arun(documents=[Document.example()])
    nodes = pipeline.run(show_progress=True, documents=documents, num_workers=1)
    #print(nodes)
    return nodes

nodes = create_nodes(documents=documents)

print('finishd nodes')


index = VectorStoreIndex.from_vector_store(vector_store, embed_model=ollama_embedding)





query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)
q = "What hosts did Pangolin persist on using a service?"
#q = "what color is the ball?"
r = query_engine.query(q)
print(r)


def test_llm(text: str = "Who is Paul Graham?"):
    '''
    Sends text to an LLM and prints the response

            Parameters:
                    text (str): The prompt

            Returns:
                    resp (str): The response from the LLM
    '''
    resp = llm.complete(text)
    print(resp)

def main():
    #test_llm("How many 'r's in strawberry?")
    print('hi')

if __name__ == "__main__":
    main()


'''
pass_embedding = ollama_embedding.get_text_embedding_batch(
    ["This is a passage!", "This is another passage"], show_progress=True
)
print(pass_embedding)

query_embedding = ollama_embedding.get_query_embedding("Where is blue?")
print(query_embedding)
'''