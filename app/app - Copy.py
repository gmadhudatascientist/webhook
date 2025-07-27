from fastapi import FastAPI, Request 
from pydantic import BaseModel, Field
from typing import Literal
import os
import faiss
import logging
from langchain_core.messages import AIMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

app = FastAPI()

logging.basicConfig(level=logging.INFO)

# ✅ Set Google API Key (for Gemini only)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyD_QkHiMP6SywCbji47EYeYEY1ysIDC00Y")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ✅ Initialize LLM
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# ✅ Use existing local PDF folder
PDF_DIR = "./downloaded_pdfs"
if not os.path.exists(PDF_DIR):
    raise FileNotFoundError(f"{PDF_DIR} not found. Please add PDF files to this folder.")

# ✅ Load and split documents
loader = PyPDFDirectoryLoader(PDF_DIR)
documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

all_splits = []
for doc in documents:
    filename = doc.metadata.get('source', 'unknown')
    splits = text_splitter.split_documents([doc])
    for split in splits:
        split.metadata['source'] = filename
    all_splits.extend(splits)

# ✅ Embedding and retriever setup
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)
vector_store = FAISS(embeddings, index, InMemoryDocstore(), {})
_ = vector_store.add_documents(all_splits)

retriever = vector_store.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_FMLA_Policy",
    "Search and return relevant information from internal Fairview policy documents."
)

# ✅ LangGraph Nodes
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevance: 'yes' or 'no'")

def generate_query_or_respond(state: MessagesState):
    system_prompt = "Always try to use the retrieval tool to answer the user's question."
    response = llm.bind_tools([retriever_tool]).invoke([
        {"role": "system", "content": system_prompt},
        *state["messages"]
    ])
    return {"messages": [response]}

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = (
        f"You are a grader assessing document relevance.\n"
        f"Document:\n{context}\n"
        f"Question: {question}\n"
        f"Relevant? Answer 'yes' or 'no'."
    )
    response = llm.with_structured_output(GradeDocuments).invoke([
        {"role": "user", "content": prompt}
    ])
    return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

def rewrite_question(state: MessagesState):
    question = state["messages"][0].content
    prompt = f"Improve this question:\n{question}"
    response = llm.invoke([{ "role": "user", "content": prompt }])
    return {"messages": [{"role": "user", "content": response.content}]}

def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    if question.strip().lower() in ["who is eligible for fmla", "who is eligible for fmla?"]:
        prompt = (
            f"You're a compliance assistant. "
            f"Give answer like this 'you must work for a covered employer, you have 12 months of employment, 1250 hours worked in past year and work at a site with 50+ employees within 75 miles'"
            f"Context:\n{context}"
        )
    #Detect if it's about eligibility, and switch to tighter phrasing
    elif "eligible" in question or "eligibility" in question:
        prompt = (
            f"You're a compliance assistant. "
            f"Give the answer in exactly **2 concise, factual sentences**. "
            f"⚠️ Do NOT use phrases like 'Based on the text' or 'According to the document'.\n\n"
            f"Context:\n{context}"
        )
    else:
        prompt = (
            f"Question: {question}\nContext: {context}\nAnswer in 2 sentences"
            f"Do NOT use phrases like 'Based on the text' or 'According to the document'.\n\n"
            f"Give the answer in exactly **2 concise, factual sentences**. "
            f"Summarize what an eligible employee is entitled to under the FMLA based on the following policy."
        )
    response = llm.invoke([{ "role": "user", "content": prompt }])
    return {"messages": [response]}

# ✅ Build LangGraph
graph = StateGraph(MessagesState)
graph.add_node(generate_query_or_respond)
graph.add_node("retrieve", ToolNode([retriever_tool]))
graph.add_node(rewrite_question)
graph.add_node(generate_answer)

graph.add_edge(START, "generate_query_or_respond")
graph.add_conditional_edges("generate_query_or_respond", tools_condition, {
    "tools": "retrieve",
    END: END,
})

graph.add_conditional_edges("retrieve", grade_documents, {
    "generate_answer": "generate_answer",
    "rewrite_question": "rewrite_question",
})
graph.add_edge("generate_answer", END)
graph.add_edge("rewrite_question", "generate_query_or_respond")

workflow = graph.compile()

# ✅ Dialogflow-compatible input
class DialogflowCXInput(BaseModel):
    query: str

@app.post("/webhook")
async def dialogflow_webhook(request: Request):
    body = await request.json()

    user_query = body.get("sessionInfo", {}).get("parameters", {}).get("user_input", "")
    if not user_query:
        user_query = body.get("text", "") or "No query found"

    messages = [{"role": "user", "content": user_query}]
    result = None

    for chunk in workflow.stream({"messages": messages}):
        for _, update in chunk.items():
            #result = update["messages"][-1].content
            last_msg = update["messages"][-1]
            if isinstance(last_msg, AIMessage):
                result = last_msg.content
            elif isinstance(last_msg, dict):
                result = last_msg.get("content", "")
            else:
                result = str(last_msg)


    return {
        "fulfillmentResponse": {
            "messages": [{
                "text": {"text": [result]}
            }]
        }
    }