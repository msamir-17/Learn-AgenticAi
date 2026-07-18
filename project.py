# SEQUENTIIIAL WORKFLOW
# ONE TIME NOTES
# So our sequntial workflow is completes but its not only about the workflow we have also completed :-
# - State
# - Nodes(Python Function)
# - StateGraph(The Graph Container)
# - Edges (Normal edges)
# - Compile and invoke 
# - How data flows through State 

import os
from typing import TypedDict

#lets create the state first
class pipelinestate(TypedDict):
    raw_input : str
    edited_text : str
    script_text : str
    final_output : str

    # this all are the state of our project 
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7 )

# first node of langGraph
def editor_node(state: pipelinestate) -> dict:
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

# second node of langGraph
def scriptwriter_node(state: pipelinestate) -> dict:
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
# third node of langGraph
def translator_node(state: pipelinestate) -> dict:
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

# Now our States and Nodes are Ready and Now its time to create the Graph
# To creating the graph we have to connect these nides and for that we have to used the edges 
# Edeges Are Very V important to create the wrokflows

from langgraph.graph import StateGraph , START , END

# Now creating the graph
graph = StateGraph(pipelinestate)

# Now adding the Nodes in our Graph
graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",scriptwriter_node)
graph.add_node("translator",translator_node)

#Add edeges ( connect two things one after another sequential )

#here is a problem which node comes first where we start 
# to fix this problem we used in build langgraph method called as START

graph.add_edge(START,"editor" ) 
graph.add_edge("editor", "scriptwriter") 
graph.add_edge("scriptwriter","translator") 
graph.add_edge("translator",END) 

#Now we are going to compile this all Nodes and Edges in one graph
app = graph.compile()

# now 'app' become runable
result = app.invoke({
    "raw_input" : "AI agent are the futiure of tech. they can think , plan and act on their own. LnagGraph helps you to build thesse agents "
})
print("your result are : - \n\n")
print(result["final_output"])