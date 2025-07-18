
from tools.reddit_api import search_reddit
from tools.serper_api import search_web
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import torch
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
    try:
        max_context_length = 1000
        if  len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            print(f"Context truncated to {max_context_length} characters")
        inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_text = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return generated_text
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        return f"Error generating fix: {str(e)}"



def gather_web_fixes(query):
    try:
        print(f"Searching for: {query}")
        
        reddit_results = []
        web_results = []
        
        # Try Reddit search with error handling
        try:
            reddit_results = search_reddit(query) or []
            print(f"Reddit results: {len(reddit_results)}")
        except Exception as e:
            print(f"Reddit search failed: {e}")
            
        # Try web search with error handling
        try:
            web_results = search_web(query) or []
            print(f"Web results: {len(web_results)}")
        except Exception as e:
            print(f"Web search failed: {e}")

        all_results = reddit_results + web_results
        
        # Better filtering
        filtered_results = []
        for res in all_results:
            if res and isinstance(res, dict):
                title = res.get('title', 'No title')
                content = res.get('content', 'No content')
                if title and content:
                    filtered_results.append({'title': title, 'content': content})

        print(f"Filtered results: {len(filtered_results)}")
        
        if not filtered_results:
            return "No relevant information found from web sources."
        
        # Combine results with length limit
        combined_text = ""
        for res in filtered_results:
            addition = f"Title: {res['title']}\nContent: {res['content']}\n\n"
            if len(combined_text + addition) > 2000:  # Limit total context
                break
            combined_text += addition

        return combined_text if combined_text else "No relevant information found."
        
    except Exception as e:
        print(f"Error in gather_web_fixes: {e}")
        
        return f"Error gathering information: {str(e)}"




def generate_fix(query):
    context = gather_web_fixes(query)
    print("Gathered Context:\n", context)
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
    
    if not fix or len(fix.strip()) < 20:
        print("Not storing fix: too short or empty")
        return 
    text = f"Problem: {query}\nFix: {fix}"

    if vector_db is None:
        vector_db = FAISS.from_texts([text], embedding)
    else:
        vector_db.add_texts([text])
    
    vector_db.save_local(vector_store_path)
    print("fix stored successfully")

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
    if result and len(result.strip()) > 20 and "Fix:" in result:
        state["final_fix"] = f"Found similar fix:\n{result}"
        print("Routing to end (found existing fix)")
        return "end"
    else:
        print("Routing to gather (no existing fix found)")
        return "gather"

def check_vector_store(state: FixAgentState):
    # This function now just calls the routing function
    return state
    
def gather_all_info(state: FixAgentState):
    print("Gathering information from web sources...")
    context = gather_web_fixes(state["user_query"])
    state["observations"].append(context)
    print(f"Added context to observations: {len(context)} characters")
    return state

def generate_fix_llm(state: FixAgentState):
    print("Generating fix using LLM...")
    context = state["observations"][-1] if state["observations"] else ""
    
    if not context:
        print("Warning: No context available for LLM")
        context = "No additional information available."
    
    prompt = f"The user has the following laptop issue: {state['user_query']}\n\nBased on the information below, provide a detailed fix:\n\n{context}\n\nFix:"
    
    fix = get_fix_from_llm(prompt)
    state["final_fix"] = fix
    print(f"Generated fix: {fix[:200]}...")
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

