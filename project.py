import os
from typing import TypedDict

#lets create the state first
def pipelinestate(TypedDict):
    raw_input : str
    edited_text : str
    script_text : str
    final_output : str

    # this all are the state of our project 
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatil", temperature=0.7 )

# first node of langGraph
def editor(state: pipelinestate) -> dict:
    # here we first write a docString. it means it tell the work of this fuction 
    """Stage 1: Clean up grammar, removes  typos, and refines the tone. """
    prompt = (
        "You are an export copyeditor. clean up the following raw text. "
        "fixany grammatical errors, spelling mistakes, and smooth out the transition flow "
        "while keeping the core message intact. Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}" 
    )
    response = llm.invoke(prompt)

    return {"edited_text" : response.content.strip()}

def editor(state: pipelinestate) -> dict:
    # here we first write a docString. it means it tell the work of this fuction 
    """Stage 2: Format the clean text into an engaging video script style"""
    print("\n--- [Stage 2] Executing ScriptWriter  Node ---")
    prompt = (
        "You are a charismatic Youtube content creator. take this edited text and transform"
        "it into a highly engaging, punchy, conversational video script hook. make it sound"
        "like a real person speaking passionately. return only the script content. \n\n"
        f"Text:\n{state['edited_text']}" 
    )
    response = llm.invoke(prompt)

    return {"script_text" : response.content.strip()}

def editor(state: pipelinestate) -> dict:
    """Stage 3: Translates the script into natural flowing hinglish."""
    print("\n--- [Stage 3] Executing Hinglish translator Node ---")

    prompt = (
        "You are an export Content localizer for the indian market. take the following script "
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
        "or repeat inportmation. alternating comfortably between hindi and english "
        "an intellectual tech educator would speak naturally on a live stream. keep the energy"
        f"Text:\n{state['script_text']}" 
    )
    response = llm.invoke(prompt)

    return {"final_output" : response.content.strip()}