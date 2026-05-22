import os
import chainlit as cl
from collections import deque
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document

from config import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    MODEL_NAME, 
    EMBEDDING_MODEL, 
    CHROMA_PATH, 
    BM25_PATH
)
from core.storage import PersistentStorage
from core.query_classifier import QueryClassifier
from core.greeting_handler import GreetingHandler
from core.enhanced_retriever import EnhancedRetriever
from core.rag_chain import EnhancedRAGChain
from models.configs import RetrievalConfig
from utils.document_loader import DocumentLoader
from utils.logging_config import setup_logging

# Set up logging
logger = setup_logging()


# --- Simple Chat Memory (replaces deprecated ConversationBufferWindowMemory) ---
class SimpleChatMemory:
    """Lightweight chat memory that stores last K messages."""

    def __init__(self, k=10):
        self.messages = deque(maxlen=k * 2)  # k pairs of user/ai messages
        self.chat_memory = self  # Self-reference for compatibility

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_ai_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def get_history_string(self) -> str:
        if not self.messages:
            return ""
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)

    @property
    def messages_list(self):
        return list(self.messages)

@cl.on_chat_start
async def on_chat_start():
    """Initialize RAG system components on chat start"""
    try:

        # Initialize storage
        storage = PersistentStorage(CHROMA_PATH, BM25_PATH)
        
        # Initialize embeddings and LLM
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        llm = ChatOllama(model=MODEL_NAME)
        
        # Initialize memory
        memory = SimpleChatMemory(k=10)
        if memory is None:
            raise ValueError("Failed to initialize SimpleChatMemory.")        
        if not storage.storage_exists():
            # Request file upload
            files = await cl.AskFileMessage(
                content="No existing embeddings found. Please upload PDF files to begin!",
                accept=['application/pdf'],
                max_files=10,
                max_size_mb=200,
                timeout=180
            ).send()
            
            if files:
                # Load and process documents
                documents = await DocumentLoader.load_pdf_documents(
                    [file.path for file in files], 
                    chunk_size=CHUNK_SIZE, 
                    chunk_overlap=CHUNK_OVERLAP
                )
                
                # Create vector store
                vector_store = Chroma.from_documents(
                    documents, 
                    embeddings,
                    persist_directory=CHROMA_PATH
                )
                vector_store.persist()
                
                # Create retrievers
                vector_retriever = vector_store.as_retriever(search_kwargs={'k': 6})
                bm25_retriever = BM25Retriever.from_documents(documents)
                
                # Initialize RAG components
                config = RetrievalConfig()
                classifier = QueryClassifier(llm)
                greeting_handler = GreetingHandler(llm)
                enhanced_retriever = EnhancedRetriever(
                    vector_retriever, 
                    bm25_retriever, 
                    llm, 
                    config
                )
                rag_chain = EnhancedRAGChain(
                    enhanced_retriever, 
                    classifier, 
                    greeting_handler, 
                    memory, 
                    llm
                )
                
                # Store in session
                cl.user_session.set("rag_chain", rag_chain)
                cl.user_session.set("memory", memory)
                
                await cl.Message(content="Embeddings created and stored. Ready to answer your questions!").send()
        else:
            # Load existing embeddings
            vector_store = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings
            )
            vector_retriever = vector_store.as_retriever(search_kwargs={'k': 6})

            # Properly retrieve documents
            documents = [
                Document(page_content=doc, metadata={"source": "existing"}) 
                for doc in vector_store.get()['documents']
            ]
            bm25_retriever = BM25Retriever.from_documents(documents)
            
            # Initialize RAG components
            config = RetrievalConfig()
            classifier = QueryClassifier(llm)
            greeting_handler = GreetingHandler(llm)
            enhanced_retriever = EnhancedRetriever(
                vector_retriever, 
                bm25_retriever, 
                llm, 
                config
            )
            rag_chain = EnhancedRAGChain(
                enhanced_retriever, 
                classifier, 
                greeting_handler, 
                memory, 
                llm
            )
            
            # Store in session
            cl.user_session.set("rag_chain", rag_chain)
            cl.user_session.set("memory", memory)
            
            await cl.Message(content="Loaded existing embeddings. Ready to answer your questions!").send()
    
    except Exception as e:
        logger.error(f"Error during chat start: {e}")
        memory = SimpleChatMemory(k=10)
        await cl.Message(content=f"Initialization error: {e}").send()

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages"""
    try:
        # Retrieve RAG chain and memory from session
        rag_chain = cl.user_session.get("rag_chain")
        memory = cl.user_session.get("memory")

        # Fallback if memory is None
        if memory is None:
            memory = SimpleChatMemory(k=10)
        # Get chat history from memory
        chat_history = memory.messages_list if hasattr(memory, 'messages_list') else []
        
        # Process query
        result = await rag_chain.process_query(
            query=message.content,
            context=str(chat_history) if chat_history else None
        )
        
        if isinstance(result['answer'], str):
            response_content = f"Query Type: {result['query_type'].value.upper()}\n\n{result['answer']}"
        else:
            response_content = f"Query Type: {result['query_type'].value.upper()}\n\n{result['answer'].content}"
            
        source_elements = []
        if result['sources']:
            for i, doc in enumerate(result['sources']):
                if isinstance(doc, str):  # Convert string to Document
                    doc = Document(page_content=doc)
                if hasattr(doc, 'page_content'):
                    source_elements.append(
                        cl.Text(
                            name=f"Source {i+1}", 
                            content=doc.page_content
                        )
                    )
            # source_elements = [
            #     cl.Text(
            #         name=f"Source {i+1}", 
            #         content=doc.page_content
            #     ) for i, doc in enumerate(result['sources'])
            # ]
            source_elements.extend(source_elements)
        
        # Send response
        await cl.Message(
            content=response_content,
            elements=source_elements if source_elements else None
        ).send()
        
        # Update memory
        memory.chat_memory.add_user_message(message.content)
        memory.chat_memory.add_ai_message(result['answer'])
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await cl.Message(content=f"Processing error: {e}").send()

if __name__ == "__main__":
    # Ensure storage directories exist
    os.makedirs(CHROMA_PATH, exist_ok=True)
    os.makedirs(BM25_PATH, exist_ok=True)
    
    # Run Chainlit app
    cl.run(port=8000)