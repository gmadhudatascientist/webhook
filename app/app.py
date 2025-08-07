# ✅ Refined LangGraph QA Workflow with Gemini Flash (Enhanced Accuracy for Eligibility Queries)

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Literal
import os
import faiss
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

# ✅ App setup
app = FastAPI()

# ✅ Google API Key and Gemini LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyD_QkHiMP6SywCbji47EYeYEY1ysIDC00Y")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai", config={"temperature": 0})

# ✅ PDF loading and sentence-level chunking
loader = PyPDFDirectoryLoader("./downloaded_pdfs")
documents = loader.load()

# Sentence-level splitting to improve fine-grained retrieval
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=50)
all_splits = []
for doc in documents:
    filename = doc.metadata.get("source", "unknown")
    splits = text_splitter.split_documents([doc])
    for split in splits:
        split.metadata['source'] = filename
    all_splits.extend(splits)

# ✅ Embedding + FAISS setup
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
index = faiss.IndexFlatL2(len(embeddings.embed_query("test")))
vector_store = FAISS(embeddings, index, InMemoryDocstore(), {})
_ = vector_store.add_documents(all_splits)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})
retriever_tool = create_retriever_tool(retriever, "retrieve_fairview_info", "Search Fairview policy and service information")

# ✅ LangGraph logic
class GradeDocuments(BaseModel):
    binary_score: str

def generate_query_or_respond(state: MessagesState):
    original_question = state["messages"][0].content.lower().strip()

    # Rewriting known variants to boost match rate
    reworded_question = original_question
    if "eligible for an evisit" in original_question:
        reworded_question = "who can use evisit"
    elif "who is eligible for fmla" in original_question:
        reworded_question = "what is the fmla policy"
    elif "am i eligible if i work part-time" in original_question:
        reworded_question = "can part-time employees get fmla"
    elif "what happens when fmla ends" in original_question:
        reworded_question = "what are employee rights after fmla ends"
    elif "Payment required" in original_question:
        reworded_question = "Is Payment required upfront for evisit"
    
    elif "need to download anything" in original_question:
            reworded_question = "do i need to download any app for an evisit"

    reworded_message = {"role": "user", "content": reworded_question}
    system_prompt = "Always use the retriever tool to find relevant answers from internal documentation."
    return {"messages": [
        llm.bind_tools([retriever_tool]).invoke([
            {"role": "system", "content": system_prompt},
            reworded_message
        ])
    ]}

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = f"Given this document excerpt:\n{context}\n\nCan it help answer the question: '{question}'? Reply with 'yes' or 'no'."
    result = llm.with_structured_output(GradeDocuments).invoke([{"role": "user", "content": prompt}])
    return "generate_answer" if result.binary_score.strip().lower() == "yes" else "rewrite_question"

def rewrite_question(state: MessagesState):
    original = state["messages"][0].content
    prompt = f"Rephrase this question to better match official policy or FAQ language:\n'{original}'"
    revised = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": revised.content}]}

def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = f"You are a Fairview chatbot.\nQuestion: {question}\nContext: {context}\n\nGive a direct, clear 1-2 sentence answer. Do not say 'according to the document' and 'based on the information provided' '."
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# ✅ LangGraph build
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
graph.add_edge("rewrite_question", "generate_query_or_respond")
graph.add_edge("generate_answer", END)
workflow = graph.compile()

# ✅ FastAPI Webhook for Dialogflow
class DialogflowCXInput(BaseModel):
    query: str

# @app.post("/webhook")
# async def dialogflow_webhook(request: Request):
#     body = await request.json()
#     user_query = body.get("sessionInfo", {}).get("parameters", {}).get("user_input", "") or body.get("text", "")
#     messages = [{"role": "user", "content": user_query}]
#     result = ""

#     for chunk in workflow.stream({"messages": messages}):
#         for _, update in chunk.items():
#             last_msg = update["messages"][-1]
#             if isinstance(last_msg, AIMessage):
#                 result = last_msg.content
#             elif isinstance(last_msg, dict):
#                 result = last_msg.get("content", "")
#             else:
#                 result = str(last_msg)

#     return {
#         "fulfillmentResponse": {
#             "messages": [{"text": {"text": [f"Fairview Chatbot(RAG): {result}"]}}]
#         }
#     }
import asyncio
from fastapi.responses import JSONResponse

@app.post("/webhook")
async def dialogflow_webhook(request: Request):
    body = await request.json()
    user_query = body.get("sessionInfo", {}).get("parameters", {}).get("user_input", "") or body.get("text", "")
    messages = [{"role": "user", "content": user_query}]
    
    async def run_workflow():
        result = ""
        async for chunk in workflow.astream({"messages": messages}):
            for _, update in chunk.items():
                last_msg = update["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    result = last_msg.content
                elif isinstance(last_msg, dict):
                    result = last_msg.get("content", "")
                else:
                    result = str(last_msg)
        return result

    try:
        # ⏱ Timeout logic (22 seconds)
        final_result = await asyncio.wait_for(run_workflow(), timeout=22.0)
    except asyncio.TimeoutError:
        final_result = "Sorry, I could answer your question but I am able to respond to FMLA policy and Company info. Please try again with other Query."

    return JSONResponse({
        "fulfillmentResponse": {
            "messages": [{"text": {"text": [f"Fairview Chatbot(RAG): {final_result}"]}}]
        }
    })
