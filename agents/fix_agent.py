
from tools.reddit_api import search_reddit
from tools.stack_overflow import search_stackoverflow
from tools.serper_api import search_web
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from typing import List, Optional, TypedDict
from langgraph.graph import StateGraph

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token


class FixAgentState(TypedDict):
    user_query: str
    thoughts: List[str]
    selected_tool: Optional[str]
    observations: List[str]
    final_fix: Optional[str]

def get_fix_from_llm(context: str) -> str:
    tokenizer.pad_token = tokenizer.eos_token  # 👈 Fix for padding issue
    inputs = tokenizer(context, return_tensors="pt", padding=True)
    output = model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(output[0], skip_special_tokens=True)



def gather_web_fixes(query):
    reddit_results = search_reddit(query) or []
    stack_results = search_stackoverflow(query) or []
    web_results = search_web(query) or []

    all_results = reddit_results + stack_results + web_results

    # Filter out any result that is missing "title" or "content"
    filtered_results = [
        res for res in all_results if res and "title" in res and "content" in res
    ]

    combined_text = "\n".join(
        f"{res['title']}\n{res['content']}" for res in filtered_results
    )

    return combined_text



def generate_fix(query):
    context = gather_web_fixes(query)
    prompt = f"The user has the following laptop issue : {query}\nBased on the information below, provide a fix or troubleshooting guide: \n\n{context}"
    return get_fix_from_llm(prompt)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store_path = "vector_store"
vector_db = None

def get_vector_db():
    global vector_db
    if vector_db is None:
        if os.path.exists(os.path.join(vector_store_path, "index.faiss")):
            vector_db = FAISS.load_local(vector_store_path, embedding, allow_dangerous_deserialization=True)
    return vector_db

def store_fix(query, fix):
    global vector_db
    text = f"Problem: {query}\nFix: {fix}"
    
    if vector_db is None:
        # Create vector store with first text
        vector_db = FAISS.from_texts([text], embedding)
    else:
        # Add to existing vector store
        vector_db.add_texts([text])
    
    vector_db.save_local(vector_store_path)

def search_vector_store(query):
    vector_db = get_vector_db()
    if vector_db is None:
        return None
    
    results = vector_db.similarity_search(query, k=1)
    return results[0].page_content if results else None

def fix_agent_main(user_query):
    previous_fix = search_vector_store(user_query)
    if previous_fix:
        return f"Found similar fix in memory: \n{previous_fix}"
    
    fix = generate_fix(user_query)
    print("Fix Generated", fix)
    return fix

def route_after_check(state: FixAgentState):
    result = search_vector_store(state["user_query"])
    if result:
        state["final_fix"] = f"Found similar fix:\n{result}"
        return "end"
    else:
        return "gather"

def check_vector_store(state: FixAgentState):
    # This function now just calls the routing function
    return state
    
def gather_all_info(state: FixAgentState):
    context = gather_web_fixes(state["user_query"])
    state["observations"].append(context)
    return state

def generate_fix_llm(state: FixAgentState):
    context = state["observations"][-1] if state["observations"] else ""
    prompt = f"the user has the following laptop issue : {state['user_query']}\nBased on the information below, provide a fix: \n\n{context}"
    fix = get_fix_from_llm(prompt)
    state["final_fix"] = fix
    return state

def store_fix_node(state: FixAgentState):
    store_fix(state["user_query"], state["final_fix"])
    return state

def return_final_fix(state: FixAgentState):
    return state

def llm_think(state: FixAgentState):
    thought = f"What source should I use to solve: {state['user_query']}?"
    state["thoughts"].append(thought)
    return state

workflow = StateGraph(FixAgentState)

workflow.add_node("llm_think", llm_think)
workflow.add_node("check_vector", check_vector_store)
workflow.add_node("gather", gather_all_info)
workflow.add_node("generate_fix", generate_fix_llm)
workflow.add_node("store_fix", store_fix_node)
workflow.add_node("end", return_final_fix)

workflow.set_entry_point("check_vector")

workflow.add_conditional_edges(
    "check_vector",
    lambda state: route_after_check(state),
    {
        "end": "end",
        "gather": "gather"
    }
)
workflow.add_edge("gather", "generate_fix")
workflow.add_edge("generate_fix", "store_fix")
workflow.add_edge("store_fix", "end")

app = workflow.compile()

query = "laptop not turning on after charging overnight"
state = FixAgentState(
    user_query=query,
    thoughts=[],
    selected_tool=None,
    observations=[],
    final_fix=None
)

final_result = app.invoke(state)
print("Final Fix:", final_result["final_fix"])