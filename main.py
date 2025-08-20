import os
from dotenv import load_dotenv
import asyncio

# LLM imports
from llama_index.llms.ollama import Ollama

# Embedding and vector store imports
import chromadb
#from chromadb.utils.batch_utils import create_batches
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex
import itertools
#import uuid

# Embedding pipeline imports
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader

# Add some tools for agents
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)

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

SYSTEM_PROMPT_RAG_AGENT="""
You are a helpful assistant specializing in cybersecurity and incident response.
You have access to a database containing data from an intrusion.
"""

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
# Set up funtions
#####
def read_data():
    """
    Load files from a directory and return a list of documents 
    """
    print(f"Loading documents from {DATA_DIR}")
    reader = SimpleDirectoryReader(input_dir=DATA_DIR,recursive=True)
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents")
    return (documents)

def create_nodes(documents, pipeline):
    """
    Create nodes. Need to be provided a LlamaIndex ingestion pipeline 
    and a list of documents (e.g. from read_data)
    """
    nodes = pipeline.run(show_progress=True, documents=documents, num_workers=1)
    #nodes = await pipeline.arun(documents=[Document.example()])
    #print(nodes)
    return nodes


'''
ChromaDB has a max batch size of 5461 documents in a given batch
Our transformations in the IngestionPipeline break one single document up into 
multiple documents. So even if you only have 100 files you read in with
SimpleDirectoryReader, you can still exceed the ChromaDB max batch size depending on
how you chunk up each file in that directory with SentenceSplitter.
The `show_progress=True` arg in pipeline.run() gives you a lazy way to see
how many documents you are processing in a given batch.

This function gives you a dirty way to chunk up the documents.
'''
def chunk_iterate(lst: list, chunk_size: int, pipeline):
    """
    Batches the documents to manage ChromaDB's 5461 max document limit
    """
    if chunk_size < len(lst):
        it = itertools.islice(lst,0,None)
        while True:
            chunk = list(itertools.islice(it, chunk_size))
            if not chunk:
                break
            create_nodes(chunk, pipeline=pipeline)
    else:
        raise RuntimeError(f"Chunk size ({chunk_size}) needs to be shorter than the length of the list ({len(lst)})")


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


#####
# Set prerequisites for embedding (rag creation and retrieval)
#####

# Set up embedding
ollama_embedding = OllamaEmbedding(
    model_name=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
    ollama_additional_kwargs={"mirostat": 0},
)

# Create chroma DB
db = chromadb.PersistentClient(path="./alfred_chroma_db")

chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=ollama_embedding)

# Create ingestion pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=2048, chunk_overlap=40),
        #HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ollama_embedding,
    ],
    vector_store=vector_store,
)

query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
    similarity_top_k=10
)

#####
# Set up reqs for the Agent
#####

# This is a tool that is used by an agent
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="RAG Query",
    description="Queries the RAG of cyber intrusion data",
    return_direct=False,
)

#This is an agent. It can be run with the .run() method.
# Because it uses the .from_tools_or_functions() method
# it creates an agent from just a tool or function
query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt=SYSTEM_PROMPT_RAG_AGENT
)


agent2 = FunctionAgent(
    tools=[query_engine_tool],
    llm=llm,
    system_prompt=SYSTEM_PROMPT_RAG_AGENT,
)

#This is also an agent. It can be run with the .run() method
# However, it can't just accept a tool. It needs an agent ()
agent = AgentWorkflow(
    #agents=[calculator_agent, query_agent], root_agent="calculator"
    #agents=[query_engine_agent]
    #agents=[query_engine_tool]
    #agents=[agent2]
    tools=[query_engine_tool]
)

# a tool (function) for an agent
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b
# an agent that uses a function
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic multiplication",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    #Note that it uses the function name, and can use it as a tool with no extra work
    tools=[multiply],
    llm=llm,
)


async def run_agent(query):
    response = await agent.run(user_msg=query)
    #response = await calculator_agent.run(user_msg=query)
    #response = await agent2.run(user_msg=query)
    #response = await query_engine_agent.run(user_msg=query)
    return response

def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, help="Pass a quoted string to test the LLM is working properly")
    parser.add_argument('--rag', action='store_true', help="Create RAG vector DB")
    parser.add_argument('--chunk_rag', type=int, help="Chunk the documents in groups of `n` documents")
    parser.add_argument('--query', type=str, help="Pass a quoted string to query the RAG")
    parser.add_argument('--agent',type=str, help="Pass a string to provide to the agent.")

    args = parser.parse_args()

    if args.test:
        text = args.test
        print(f"Sending request of:\n {text}\n to model: {LLM_MODEL}\n at: {OLLAMA_BASE_URL}")
        #test_llm("How many 'r's in strawberry?")
        test_llm(text)

    elif args.rag:
        documents = read_data()
        nodes = create_nodes(documents=documents, pipeline=pipeline)
        print('finished nodes')

    elif args.query:
        q = args.query
        print(q)
        r = query_engine.query(q)
        print(r)
    
    elif args.chunk_rag:
        chunk_rag = args.chunk_rag
        documents = read_data()
        chunk_iterate(lst=documents, chunk_size=chunk_rag, pipeline=pipeline)
    
    elif args.agent:
        q = args.agent
        #pass
        r = asyncio.run(run_agent(q))
        #response = await agent.run(user_msg=q)
        print(r)


if __name__ == "__main__":
    main()