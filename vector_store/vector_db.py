from transformers import AutoTokenizer, AutoModelForCasualLM
from tools.reddit_api import search_reddit
from tools.stack_overflow_api import search_stack_overflow
from tools.serper_api import search_serper
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from typing import List, 
from langgraph.graph import State
from langgrapg.graph import StateGraph

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True)
model = AutoModelForCasulaLM.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True).eval()

def get_fix_from_llm(context):
    input_ids = tokenizer.encode(context, return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens=300)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def gather_web_fixes(query):
    reddit_results = search_reddit(query)
    stack_results = search_stack_overflow(query)
    web_results = search_serper(query)

    all_results = reddit_results + stack_results + web_results
    combined_text = "\n".join([res["title"] + "\n" + res["content"] for res in all_results])
    return combined_text

def generate_fix(query):
    context = gather_web_fixes(query)
    prompt = f"The user has the following laptop issue : {query}\nBased on the information below, provide a fix or troubleshooting quide: \n\n{context}"
    return get_fix_from_llm(prompt)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("vector_store",embedding) if os.path.exists("vector_store") else FAISS.from_texts([], embedding)

def store_fix(query, fix):
    vector_db.add_texts([f"Problem: {query}\nFix: {fix}"])
    vector_db.save_local("vector_store")

def search_vector_store(query):
    results = vector_db.similarity_search(query, k=1)
    return results[0].page_content if results else None

def fix_agent_main(user_query):
    previous_fix = search_vector_store(user_query)
    if previous_fix:
        return f"Found similar fix in memory: \n{previous_fix}"
    
    fix = generate_fix(user_query)
    print("Fix Generated", fix)

    return fix
def check_vector_store(state: FixAgentState):
    result = search_vector_store(state.user_query)
    if result:
        state.final_fix = f"Found similar fix:\n{result}"
        return "end"
    else:
        return "gather"
    
def gather_all_info(state: FixAgentState):
    context = gather_web_fixes(state.user_query)
    state.observations.append(context)
    return state

def generate_fix_llm(state: FixAgentState):
    context = state.observations[-1] if state.observations else ""
    prompt = f"the user has the following laptop issue : {state.user_query}\nBased on the information below, provide a fix: \n\n{context}"
    fix = get_fix_from_llm(prompt)
    state.final_fix = fix
    return state

def store_fix_node(state: FixAgentState):
    store_fix(state.user_query, state.final_fix)
    return state

def return_final_fix(state: FixAgentState):
    return state

class FixAgentState(State):
    user_query: str
    thoughts: List[str] = []
    selected_tool: Optional[str] = None
    observations: List[str] = []
    final_fix: Optional[str] = None

def llm_think(state: FixAgentState):
    thought = f"What source should I use to solve: {state.user_query}?"
    state.thoughts.append(thought)
    return state


workflow = StateGraph(FixAgentState)

workflow.add_node("llm_think", llm_think)
workflow.add_node("check_vector", check_vector_store)
workflow.add_node("gather", gather_all_info)
workflow.add_node("generate_fix", generate_fix_llm)
workflow.add_node("store_fix", store_fix_node)
workflow.add_node("end", return_final_fix)

workflow.set_entry_point("check_vector")

workflow.add_edge("check-vector", "gather")
workflow.add_edge("gather", "generate_fix")
workflow.add_edge("generate_fix", "store_fix")
workflow.add_edge("store_fix", "end")

app = workflow.compile()

query = "laptop not turning on after charging overnight"
state = FixAgentState(user_query = query)

final_result = app.invoke(state)
print("Final Fix:", final_result.final_fix)



    

