import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv
load_dotenv()

# Load API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Load documents
loader = TextLoader("data/sample.txt")
documents = loader.load()

# 2. Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3. Embeddings + Vector DB
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# 4. Retriever
retriever = vectorstore.as_retriever()

# 5. LLM
llm = ChatOpenAI(temperature=0)

# 6. Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 7. RAG Chat Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# 8. Chat loop
print("SmartDoc AI (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    result = qa_chain({"question": query})
    print("AI:", result["answer"])
