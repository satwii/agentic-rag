from tools.reddit_api import search_reddit
from tools.serper_api import search_web
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from typing import List, Optional, TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# Initialize the LLM components
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token

# Global vector store setup
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store_path = "vector_store"
vector_db = None

class AgentState(TypedDict):
    """State schema for the laptop repair agent"""
    user_query: str
    search_results: Optional[str]
    existing_fix: Optional[str]
    generated_fix: Optional[str]
    final_answer: Optional[str]
    step_history: List[str]
    next_action: Optional[str]

def get_vector_db():
    """Initialize or load the vector database"""
    global vector_db
    if vector_db is None:
        if os.path.exists(os.path.join(vector_store_path, "index.faiss")):
            try:
                vector_db = FAISS.load_local(
                    vector_store_path, 
                    embedding, 
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector database loaded successfully")
            except Exception as e:
                print(f"❌ Error loading vector database: {e}")
                vector_db = None
    return vector_db

def search_existing_fixes(state: AgentState) -> AgentState:
    """Node: Search for existing fixes in vector store"""
    print("🔍 Searching for existing fixes...")
    
    db = get_vector_db()
    if db is None:
        print("📝 No existing vector store found")
        state["step_history"].append("No existing knowledge base found")
        state["next_action"] = "search_web"
        return state
    
    try:
        results = db.similarity_search(state["user_query"], k=1)
        if results and len(results[0].page_content.strip()) > 20:
            existing_fix = results[0].page_content
            print(f"✅ Found existing fix: {existing_fix[:100]}...")
            
            state["existing_fix"] = existing_fix
            state["final_answer"] = f"🔄 **Found Similar Solution:**\n\n{existing_fix}"
            state["step_history"].append("Retrieved existing fix from knowledge base")
            state["next_action"] = "complete"
            return state
        else:
            print("❌ No relevant existing fixes found")
            state["step_history"].append("No relevant existing fixes found")
            state["next_action"] = "search_web"
            return state
            
    except Exception as e:
        print(f"❌ Error searching vector store: {e}")
        state["step_history"].append(f"Error searching knowledge base: {str(e)}")
        state["next_action"] = "search_web"
        return state

def search_web_sources(state: AgentState) -> AgentState:
    """Node: Search web sources (Reddit + Web) for solutions"""
    print("🌐 Searching web sources...")
    
    query = state["user_query"]
    all_results = []
    
    # Search Reddit
    try:
        reddit_results = search_reddit(query) or []
        print(f"📱 Reddit results: {len(reddit_results)}")
        all_results.extend(reddit_results)
    except Exception as e:
        print(f"❌ Reddit search failed: {e}")
    
    # Search Web
    try:
        web_results = search_web(query) or []
        print(f"🌍 Web results: {len(web_results)}")
        all_results.extend(web_results)
    except Exception as e:
        print(f"❌ Web search failed: {e}")
    
    # Process and filter results
    combined_content = ""
    valid_results = 0
    
    for result in all_results:
        if result and isinstance(result, dict):
            title = result.get('title', '').strip()
            content = result.get('content', '').strip()
            
            if title and content and len(content) > 10:
                addition = f"**{title}**\n{content}\n\n"
                if len(combined_content + addition) > 2000:
                    break
                combined_content += addition
                valid_results += 1
    
    if combined_content:
        state["search_results"] = combined_content
        state["step_history"].append(f"Found {valid_results} relevant web sources")
        state["next_action"] = "generate_fix"
        print(f"✅ Collected {valid_results} valid results")
    else:
        state["search_results"] = "No relevant information found from web sources."
        state["step_history"].append("No relevant web sources found")
        state["next_action"] = "generate_fix"
        print("❌ No valid results found")
    
    return state

def generate_solution(state: AgentState) -> AgentState:
    """Node: Generate solution using LLM"""
    print("🤖 Generating solution with LLM...")
    
    try:
        # Prepare context
        context = state.get("search_results", "No additional information available.")
        
        prompt = f"""You are a laptop repair expert. Help solve this issue:

PROBLEM: {state['user_query']}

AVAILABLE INFORMATION:
{context}

Please provide a clear, step-by-step solution. Include:
1. Possible causes
2. Troubleshooting steps
3. When to seek professional help

Solution:"""
        
        # Truncate if too long
        max_length = 1000
        if len(prompt) > max_length:
            prompt = prompt[:max_length] + "..."
            print(f"⚠️ Context truncated to {max_length} characters")
        
        # Generate with LLM
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            generated_text = tokenizer.decode(
                output[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
        
        if generated_text:
            state["generated_fix"] = generated_text
            state["final_answer"] = f"🔧 **Generated Solution:**\n\n{generated_text}"
            state["step_history"].append("Generated new solution using AI")
            state["next_action"] = "store_solution"
            print(f"✅ Generated solution: {generated_text[:100]}...")
        else:
            fallback = "Unable to generate a specific solution. Please consult a professional technician."
            state["generated_fix"] = fallback
            state["final_answer"] = f"⚠️ **Fallback Response:**\n\n{fallback}"
            state["step_history"].append("Used fallback response")
            state["next_action"] = "complete"
            print("⚠️ Used fallback response")
            
    except Exception as e:
        error_msg = f"Error generating solution: {str(e)}"
        print(f"❌ {error_msg}")
        state["generated_fix"] = error_msg
        state["final_answer"] = f"❌ **Error:**\n\n{error_msg}"
        state["step_history"].append(error_msg)
        state["next_action"] = "complete"
    
    return state

def store_solution(state: AgentState) -> AgentState:
    """Node: Store new solution in vector database"""
    print("💾 Storing solution...")
    
    try:
        global vector_db
        
        fix = state.get("generated_fix", "").strip()
        if not fix or len(fix) < 20 or fix.lower() == "no content":
            print("⚠️ Invalid or empty solution, not storing")
            state["step_history"].append("Solution not stored (invalid or empty)")
            state["next_action"] = "complete"
            return state

            print("⚠️ Solution too short, not storing")
            state["step_history"].append("Solution not stored (too short)")
            state["next_action"] = "complete"
            return state
        
        # Create document text
        doc_text = f"Problem: {state['user_query']}\nSolution: {fix}"
        
        # Initialize or update vector store
        if vector_db is None:
            vector_db = FAISS.from_texts([doc_text], embedding)
            print("🆕 Created new vector database")
        else:
            vector_db.add_texts([doc_text])
            print("➕ Added to existing vector database")
        
        # Save to disk
        os.makedirs(vector_store_path, exist_ok=True)
        vector_db.save_local(vector_store_path)
        
        state["step_history"].append("Solution stored in knowledge base")
        state["next_action"] = "complete"
        print("✅ Solution stored successfully")
        
    except Exception as e:
        error_msg = f"Error storing solution: {str(e)}"
        print(f"❌ {error_msg}")
        state["step_history"].append(error_msg)
        state["next_action"] = "complete"
    
    return state

def complete_task(state: AgentState) -> AgentState:
    """Node: Final completion"""
    print("✅ Task completed")
    state["step_history"].append("Task completed")
    return state

def route_next_step(state: AgentState) -> Literal["search_web", "generate_fix", "store_solution", "complete"]:
    """Conditional edge router"""
    next_action = state.get("next_action", "search_web")
    print(f"🔀 Routing to: {next_action}")
    return next_action

# Build the LangGraph workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("search_existing", search_existing_fixes)
workflow.add_node("search_web", search_web_sources)
workflow.add_node("generate_fix", generate_solution)
workflow.add_node("store_solution", store_solution)
workflow.add_node("complete", complete_task)

# Set entry point
workflow.set_entry_point("search_existing")

# Add conditional routing
workflow.add_conditional_edges(
    "search_existing",
    route_next_step,
    {
        "search_web": "search_web",
        "complete": "complete"
    }
)

workflow.add_conditional_edges(
    "search_web",
    route_next_step,
    {
        "generate_fix": "generate_fix"
    }
)

workflow.add_conditional_edges(
    "generate_fix",
    route_next_step,
    {
        "store_solution": "store_solution",
        "complete": "complete"
    }
)

workflow.add_conditional_edges(
    "store_solution",
    route_next_step,
    {
        "complete": "complete"
    }
)

# End at complete
workflow.add_edge("complete", END)

# Compile with memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def run_laptop_repair_agent(query: str, thread_id: str = "default"):
    """Main function to run the laptop repair agent"""
    print(f"🚀 Starting Laptop Repair Agent")
    print(f"Query: {query}")
    print("=" * 60)
    
    # Initial state
    initial_state = AgentState(
        user_query=query,
        search_results=None,
        existing_fix=None,
        generated_fix=None,
        final_answer=None,
        step_history=[],
        next_action=None
    )
    
    # Configuration for thread
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Run the workflow
        final_state = app.invoke(initial_state, config)
        
        print("=" * 60)
        print("🎯 FINAL RESULT:")
        print(final_state.get("final_answer", "No solution generated"))
        print("\n📋 Process Steps:")
        for i, step in enumerate(final_state.get("step_history", []), 1):
            print(f"  {i}. {step}")
        
        return final_state
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        return None

# Example usage
if __name__ == "__main__":
    query = "laptop not turning on after charging overnight"
    result = run_laptop_repair_agent(query)