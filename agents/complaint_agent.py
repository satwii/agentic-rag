from typing import Optional, List, TypedDict
import re
from tools.serper_api import search_web
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langgraph.graph import StateGraph


# Make sure these are globally initialized or already available
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token

class ComplaintAgentState(TypedDict):
    """State schema for the complaint agent"""
    user_query: str  # The original issue
    model_name: str  # Device model (e.g., "Pavilion 15")
    company_name: str  # Brand name (e.g., "HP")
    fix_attempts: List[str]  # List of fixes already tried

    support_email: Optional[str]  # Will be filled after search
    email_body: Optional[str]  # Will be filled by LLM
    step_history: List[str]  # Logs for tracing

import re
from tools.serper_api import search_web

def find_support_email(state: ComplaintAgentState) -> ComplaintAgentState:
    print("📧 Searching for support email...")

    # Create the search query
    query = f"{state['company_name']} {state['model_name']} laptop customer support email"
    
    try:
        results = search_web(query)
    except Exception as e:
        print(f"❌ Serper API error: {e}")
        results = []

    # Combine titles + content to extract emails
    combined_text = ""
    for r in results:
        combined_text += r.get("title", "") + "\n" + r.get("content", "") + "\n"
        
    # Find email with regex
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", combined_text)
    
    if emails:
        email_found = emails[0]
        print(f"✅ Found support email: {email_found}")
        state["support_email"] = email_found
        state["step_history"].append("Found support email from Serper API")
    else:
        fallback_email = f"support@{state['company_name'].lower()}.com"
        print(f"⚠️ No email found. Using fallback: {fallback_email}")
        state["support_email"] = fallback_email
        state["step_history"].append("Support email not found; used fallback")

    return state

def generate_support_email(state: ComplaintAgentState) -> ComplaintAgentState:
    print("✍️ Generating support email with Qwen...")

    issue = state["user_query"]
    model_name = state["model_name"]
    company = state["company_name"]
    fixes = "\n".join(f"- {f}" for f in state["fix_attempts"])

    prompt = f"""
You are a helpful assistant. Write a professional support email to {company}'s customer service team.

PRODUCT: {model_name}
ISSUE: {issue}
FIXES TRIED:
{fixes}

Politely request help or a solution.

Email:
"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        email_text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        state["email_body"] = email_text
        state["step_history"].append("Generated email using Qwen")

        print("✅ Email generated.")
    except Exception as e:
        error_msg = f"❌ Error generating email: {e}"
        print(error_msg)
        state["email_body"] = "Unable to generate email due to an error."
        state["step_history"].append(error_msg)

    return state

# Define a LangGraph StateGraph for the Complaint Agent
def build_complaint_graph():
    builder = StateGraph(ComplaintAgentState)

    # Register nodes
    builder.add_node("find_support_email", find_support_email)
    builder.add_node("generate_support_email", generate_support_email)

    # Define flow
    builder.set_entry_point("find_support_email")
    builder.add_edge("find_support_email", "generate_support_email")
    builder.set_finish_point("generate_support_email")

    # Compile the graph
    return builder.compile()

def run_complaint_agent(state: ComplaintAgentState) -> ComplaintAgentState:
    graph = build_complaint_graph()
    return graph.invoke(state)


test_state = {
    "user_query": "My laptop randomly shuts down after 10 minutes of usage.",
    "model_name": "HP Inspiron 15",
    "company_name": "HP",
    "fix_attempts": [
        "Checked for overheating",
        "Updated all drivers",
        "Ran hardware diagnostics"
    ],
    "support_email": None,
    "email_body": None,
    "step_history": []
}
final_state = run_complaint_agent(test_state)
print("📬 Final Email Generated:")
print(final_state["email_body"])

print("\n📧 Support Email Found:")
print(final_state["support_email"])

print("\n🪪 Steps Taken:")
for step in final_state["step_history"]:
    print("•", step)
