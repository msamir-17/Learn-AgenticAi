# start -> article -> thumbnailo -> Tweet Node -> End

# start  -|-> article
#         |-> thumbnailo -> End
#         |-> Tweet Node

# Project Idea :
# In this project we build ans= automated Ai Content Moderation & brand Safety pipeline using lnagGraph and groq
# Instead of processing an input text sequentially (which is slow and in-efficient), our pipeline takes any raw pieceof text whether its video script, a blog draft, or a user comment and broadcasts it tothree specialized aiagent running simultaneously in paralle.
# each agent evaluates this text froma completely different perspective and scores it on a scale from 0 to 100

# SCORE: 
# 1. The Toxicity Monitor : Scans the next for aggressive language, profanity or hate speech
# 2. The CopyRight Cop : Analyzes the text for plagarism, trademark violation, or unoriginal copy risks.
# 3. The Cultural Guide: Flags Regional sensitivities or political landmines that could offend a global audience 


import os 
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm= ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
def merge_score_dict(existing :dict, newupdate : dict ) -> dict:
    if existing is None:
        return newupdate
    return {**existing, **newupdate}
# create a state
class AnalyzerState(TypedDict):
    raw_text : str
    # culture_score : str 
    # culture_score : str 
    # culture_score : str 
    safety_score : Annotated[dict[str, int], merge_score_dict]


# Nodes

def toxicity_node(state: AnalyzerState) -> dict:
    print("\n[Branch 1] Analyzing Toxicity and Hate Speech...")

    prompt = (
        "You are an expert Content Moderation AI. Your job is to analyze the text provided below "
        "for profanity, aggressive language, hate speech, or toxicity.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Rate the text on a scale from 0 to 100 (0 = perfectly clean, 100 = highly toxic).\n"
        "2. Output ONLY the plain integer number. Do not include words, spaces, markdown, or punctuation.\n\n"
        f"Text to analyze:\n{state['raw_text']}"
    )
    
    response = llm.invoke(prompt)
    try:
        # Strip formatting like backticks or newlines just in case
        clean_content = response.content.strip().replace("`", "")
        score = int(clean_content)
    except ValueError:
        score = 0   
        
    return {"safety_score":{"toxicity_level": score}}
   
def copyright_node(state: AnalyzerState) -> dict:
    print("\n[Branch 2] Analyzing Plagiarism and Trademark Risks...")

    prompt = (
        "You are an expert Intellectual Property AI. Your job is to analyze the text provided below "
        "for plagiarism, trademark violations, unoriginal copy risks, or fair-use concerns.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Rate the text on a scale from 0 to 100 (0 = completely original/safe, 100 = high copyright risk).\n"
        "2. Output ONLY the plain integer number. Do not include words, spaces, markdown, or punctuation.\n\n"
        f"Text to analyze:\n{state['raw_text']}"
    )
    
    response = llm.invoke(prompt)
    try:
        clean_content = response.content.strip().replace("`", "")
        score = int(clean_content)
    except ValueError:
        score = 0   
        
    return {"safety_score":{"copyright_score": score}}

def cultural_node(state: AnalyzerState) -> dict:
    print("\n[Branch 3] Analyzing Cultural Sensitivities and Political Risks...")

    prompt = (
        "You are an expert Global Brand Safety AI. Your job is to analyze the text provided below "
        "for regional sensitivities, political landmines, or cultural triggers that could offend a global audience.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Rate the text on a scale from 0 to 100 (0 = completely safe/neutral, 100 = highly offensive/risky).\n"
        "2. Output ONLY the plain integer number. Do not include words, spaces, markdown, or punctuation.\n\n"
        f"Text to analyze:\n{state['raw_text']}"
    )
    
    response = llm.invoke(prompt)
    try:
        clean_content = response.content.strip().replace("`", "")
        score = int(clean_content)
    except ValueError:
        score = 0   
        
    return {"safety_score":{"cultural_score": score}}


# now start connection 

builder = StateGraph(AnalyzerState)

builder.add_node("toxicity_node",toxicity_node)
builder.add_node("copyright_node",copyright_node)
builder.add_node("cultural_node",cultural_node)
  
# now connect edges using FAN-IN and FAN-OUT

builder.add_edge(START,"toxicity_node")
builder.add_edge(START,"copyright_node")
builder.add_edge(START,"cultural_node")

builder.add_edge("toxicity_node",END)
builder.add_edge("copyright_node",END)
builder.add_edge("cultural_node",END)

app = builder.compile()



sample_script = {
    "Hey everyone! Today we are looking at Apple's new secret technology, and honestly, the designers are absolute idiots for building it this way. I totally copied their leaked blueprint anyway. We are filming this live from our new studio right on the disputed border line, and frankly, the local government here is completely illegitimate and corrupt."

}

initial_state = {
    "raw_text": sample_script,
    "safety_score": {} #inintialized as an empty dictionary

}

final_state = app.invoke(initial_state)

print(final_state["safety_score"])