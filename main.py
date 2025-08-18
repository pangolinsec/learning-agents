from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="llama3.1:latest",
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)



def main():
    #print("Hello from llamaindex!")
    resp = llm.complete("Who is Paul Graham?")
    print(resp)

if __name__ == "__main__":
    main()
