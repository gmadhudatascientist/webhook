from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Literal
import os, faiss, logging, re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# ⚙️ Logging
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# 🔐 Google Gemini Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyD_DUMMY_FOR_TESTING")

# 🔍 Load PDFs
PDF_DIR = "./downloaded_pdfs"
if not os.path.exists(PDF_DIR):
    raise FileNotFoundError("PDF directory not found.")
documents = PyPDFDirectoryLoader(PDF_DIR).load()

# ✂️ Split into sentences
splitter = SentenceTransformersTokenTextSplitter(chunk_size=1, chunk_overlap=0)
sentence_splits = []
for doc in documents:
    filename = doc.metadata.get('source', 'unknown')
    for sentence in splitter.split_documents([doc]):
        sentence.metadata["source"] = filename
        sentence_splits.append(sentence)

# 🔗 Embed & Index
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
dim = len(embeddings.embed_query("test"))
index = faiss.IndexFlatL2(dim)
vectorstore = FAISS(embeddings, index, InMemoryDocstore(), {})
vectorstore.add_documents(sentence_splits)

# 🔎 Retriever: top-5 relevant sentences
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_policy_sentences",
    "Retrieve top 5 policy sentences relevant to a user's question."
)

# 🤖 Gemini Chat Model
llm = init_chat_model("gemini-1.5-flash", model_provider="google_genai")

# 🎯 Relevance grader
class GradeDocuments(BaseModel):
    binary_score: str

def generate_query_or_respond(state: MessagesState):
    prompt = "Always use the tool to answer the user's question with supporting sentences."
    response = llm.bind_tools([retriever_tool]).invoke([
        {"role": "system", "content": prompt},
        *state["messages"]
    ])
    return {"messages": [response]}

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content
    grade_prompt = (
        f"Does this context answer the question below?\n\n"
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        f"Answer yes or no only."
    )
    result = llm.with_structured_output(GradeDocuments).invoke([
        {"role": "user", "content": grade_prompt}
    ])
    return "generate_answer" if result.binary_score == "yes" else "rewrite_question"

def rewrite_question(state: MessagesState):
    question = state["messages"][0].content
    prompt = f"Rewrite this question for better clarity: {question}"
    revised = llm.invoke([{ "role": "user", "content": prompt }])
    return {"messages": [{"role": "user", "content": revised.content}]}

def generate_answer(state: MessagesState):
    question = state["messages"][0].content.lower()
    context = state["messages"][-1].content

    prompt = (
        f"You are a legal policy assistant. Based only on the sentences below, "
        f"Answer this question in exactly 2 short factual sentences. If the context is limited, make the best possible inference from it."
        f"Do NOT use phrases like 'According to the policy' or 'Based on the text'.\n\n"
        f"Question: {question}\n\n"
        f"Top 5 Context Sentences:\n{context}"
    )
    result = llm.invoke([{ "role": "user", "content": prompt }])
    sentences = re.split(r'(?<=[.!?])\s+', result.content.strip())
    short_answer = " ".join(sentences[:2])
    return {"messages": [{"role": "assistant", "content": short_answer}]}

# 🔁 LangGraph setup
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
    "rewrite_question": "rewrite_question"
})
graph.add_edge("generate_answer", END)
graph.add_edge("rewrite_question", "generate_query_or_respond")
workflow = graph.compile()

# 🌐 Dialogflow webhook
class DialogflowCXInput(BaseModel):
    query: str

@app.post("/webhook")
async def dialogflow_webhook(request: Request):
    body = await request.json()
    user_query = body.get("sessionInfo", {}).get("parameters", {}).get("user_input", "") or body.get("text", "")
    if not user_query:
        return {"fulfillmentResponse": {"messages": [{"text": {"text": ["No input found"]}}]}}

    messages = [{"role": "user", "content": user_query}]
    result = None

    for chunk in workflow.stream({"messages": messages}):
        for _, update in chunk.items():
            msg = update["messages"][-1]
            if isinstance(msg, AIMessage):
                result = msg.content
            elif isinstance(msg, dict):
                result = msg.get("content", "")
            else:
                result = str(msg)

    return {
        "fulfillmentResponse": {
            "messages": [{"text": {"text": [result]}}]
        }
    }
