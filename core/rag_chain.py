import json
import asyncio
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from models.enums import QueryType

class EnhancedRAGChain:
    def __init__(self, retriever, classifier, greeting_handler, memory, llm):
        """
        Initialize the Enhanced RAG Chain.
        
        Args:
            retriever: Document retrieval system
            classifier: Query type classifier
            greeting_handler: Handles greeting interactions
            memory: Conversation memory
            llm: Language model
        """
        self.retriever = retriever
        self.classifier = classifier
        self.greeting_handler = greeting_handler
        self.memory = memory
        self.llm = llm
        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self):
        """Create prompt template for RAG processing"""
        system_template = """You are an expert interactive research Analyst connecting users with information.
        Use the following pieces of Reference Context to answer the user's question. 
        If you don't know the answer or the context doesn't contain it, truthfully say you don't know. DO NOT make up answers.
        
        Query Type: {query_type}
        
        === CONVERSATION HISTORY ===
        {chat_history}
        
        === REFERENCE DOCUMENTS / CONTEXT ===
        {context}
        
        === QUESTION ===
        {question}
        
        Additional Instructions based on query type:
        - For Factual queries: Provide precise, verifiable information citing the specific document name if provided
        - For Analytical queries: Break down the analysis step by step based on the retrieved documents
        - For Opinion queries: Present different viewpoints and perspectives found in the text
        - For Contextual queries: Focus closely on the Conversation History and how it relates the query to the documents
        - For Greetings: Respond warmly, acknowledge the available documents if mentioned in context, and offer help
        
        ALWAYS cite the source document names when referencing information.
        """
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

    async def process_query(self, query: str, context: str = None) -> Dict[str, Any]:
        """
        Process user query through RAG pipeline.
        
        Args:
            query (str): User's input query
            context (str, optional): Conversation context
        
        Returns:
            Dict[str, Any]: Processing result with answer, sources, and query type
        """
        query_type = await self.classifier.classify(query)
        
        # Handle greeting separately
        if query_type == QueryType.GREETING:
            greeting_response = await self.greeting_handler.handle_greeting(query)
            return {
                "answer": greeting_response,
                "sources": [],
                "query_type": query_type
            }
        
        # Retrieve documents
        docs = await self.retriever.retrieve_documents(query, query_type, context)
        
        # Generate response
        context_text = "\n\n".join(f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in docs)
        chat_hist = context if context else "None"
        
        response = await self.llm.ainvoke(
            self.prompt_template.format(
                query_type=query_type.value,
                context=context_text,
                chat_history=chat_hist,
                question=query
            )
        )

        return {
            "answer": response,
            "sources": docs,
            "query_type": query_type
        }

    async def astream_query(self, query: str, context: str = None):
        """Process user query and stream tokens."""
        print(f"[CHAIN] Processing query: {query}")
        yield {"type": "step", "content": "Starting analysis..."}
        
        # Parallelize classification and retrieval (using FACTUAL as default for concurrent retrieval)
        # This keeps the pipe warm while we wait for classification
        print("[CHAIN] Starting classification and parallel retrieval...")
        classify_task = asyncio.create_task(self.classifier.classify(query))
        
        # We start a FACTUAL retrieval by default to save time, 
        # but if classify says otherwise we might need a corrective retrieval.
        # For now, let's just do them sequentially but optimize each.
        
        query_type = await classify_task
        print(f"[CHAIN] Query classified as: {query_type}")
        
        if query_type == QueryType.GREETING:
            greeting_response = await self.greeting_handler.handle_greeting(query)
            yield {"type": "metadata", "query_type": query_type.value, "sources": []}
            yield {"type": "token", "content": greeting_response if isinstance(greeting_response, str) else greeting_response.content}
            return

        docs = await self.retriever.retrieve_documents(query, query_type, context)
        source_metadata = [
            {"content": doc.page_content[:200], "metadata": doc.metadata}
            for doc in docs
        ]
        
        yield {"type": "metadata", "query_type": query_type.value, "sources": source_metadata}
        
        # Stream response
        context_text = "\n\n".join(f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in docs)
        chat_hist = context if context else "None"
        
        print("[CHAIN] Starting LLM stream...")
        async for chunk in self.llm.astream(
            self.prompt_template.format(
                query_type=query_type.value,
                context=context_text,
                chat_history=chat_hist,
                question=query
            )
        ):
            yield {"type": "token", "content": chunk.content}