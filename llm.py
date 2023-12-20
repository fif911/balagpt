import os

from dotenv import load_dotenv

# load .env file

load_dotenv()

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# check if storage already exists
if not os.path.exists("storage"):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist()
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)

# either way we can now query the index
query_engine = index.as_query_engine()

print("DONE\n")

previous_query = []
while True:
    print("Input your prompt: ")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    query = '\n'.join(lines)
    if len(query) < 10:
        print("Please input a longer prompt")
        continue
    if previous_query:
        query = ("For more context this is the history of the chat. "
                 "Leverage it if you find it useful or just drop if it does not help you:\n"
                 + '\n'.join(previous_query) + "The question for you to answer it:\n" + query)
    print(f"The query is: {query}")
    print("Answering...")
    response = query_engine.query(query)
    print(response)
    previous_query.append(f"""
    The prompt was:
    {query}
    
    And your answer was:
    {response}
    """)
    print("\n")
