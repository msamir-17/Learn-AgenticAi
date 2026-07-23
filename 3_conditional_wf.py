# 1. Conditional edges
# 2. Router Function
# 3. LLM-based decision making
# 4. Tools and ToolNodes
# 5. Tools calling (LLM deciding which tool to use)
# 6. RAG inside LangGraph (retrieval as a conditional path)
# 7. add_messsage reducer
# 8. MessagesState (Prebuilt state for chat)

# Q. what is Conditional WF is 
# Ans. In Python we have already seen whta is conditional stm is (if , else , elif ) same as in langGraph But impppppp [ only work condition work at a time ]

# Conditional WF 
# - here the next step is decide based on a condition. The graph Looks at the current state and chooses path to take
# - In a sequential workflow the path is fixed - A- B - C, Every single time. In a conditional workflow , the path changes A -> (checck something) -> go to B or go to C depending on the answer

# ! Now Usually to create a conditional workflow you need 2 pieces
# P1 - A router Function

# A regular py function that reads the current state and returns the name of the next node
# eg = > 
# def router(state: State) -> str:
#     if state['technical'] == 'technical':
#         return "tech_page"
#     elif state['category'] == 'category':
#         return "category_page"
#     else:
#         return "general_page"
# it only makes a decision based on what i seee in the state, where should we go next

# P2 - Add_Conditional_edges
# Instead of "add_edges " (which always goestothe same node) we used "Add_Conditional_edges" (which calls the router function to decide).
# eg = builder.add_edges("A","B")
#    = builder.add_conditional_edges("A",router) // router is a function

# The Router Function can make decision in two ways:
#  1. Rule-Based (py logic) :
# we write the rules bu ourself. simple predictable, fast

#  2. LLM-based (Ai Decides)
# we ask the LLM to classify or decide, and route based on its answer. more flexible, handles ambiguity, "but slower" 

# PROJECT : COLLEGE CHATBOT

# we have 3 branches in our clg BCA , BCOM , BBA

# A student will choose one of these 3 options and then a chatbot will be activated.we can ask question to that chatbot, but the llm you are usingin that chatbot does not have knowledge of the college programme. for that we will have 2 PDFs the frist one will be for academics and the second one willbe for fee-related things. 

# 3 conditional path
#     1. Answer the question using the academic PDF (using RAG)
#     2. Answer the question using the fee PDF (using RAG)
#     3. Answer General question based on the LLMsown Knowledge
# And every conditional paths response converges to a single node 


import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph , START , END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


# building RAG System
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

def build_retriver(pdf_path : str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 800,chunk_overlap = 100 )


    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks,embeddings)

    return vectorstore.as_retriever(search_kwargs = {"k":4})

academic_retriever = build_retriver("academics_handbook.pdf")
fee_retriever = build_retriver("fee_structure.pdf")



llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# Step 2 : Creating State
class State(TypedDict):
    programme : str  # branch 
    messages : Annotated[list,add_messages] # messages 
    query_type : str # academic , fee , general (types of query )
    retrieved_context : str # targeted retived 

# step 3 - Nodes Generations
def classifier_node(state : State) -> dict:
    """Look At the latest user message and Decide which path to take"""

    last_message = state['messages'][-1].content  

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )  

    # ai decide the rule
    response = llm.invoke(prompt)
    category = response.content.strip().lower() 

    if "academic" in category:
        category = "academic"
    elif"fee" in category:
        category= " fee"
    else:
        category = "general"

    return {"query_type" : category }

def academic_rag_node(state: State) -> str:
    """Determines which path to execute next based on classification."""
    query = state["messages"][-1].content
    docs = academic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def fee_rag_node(state: State) -> str:
    """Determines which path to execute next based on classification."""
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def general_node(state: State) -> dict:
    """Answer directly using the llm own knowledge, no retrieval needed."""
    return {"retrieved_context" : "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    
    # Safely checks for 'programme' or falls back to 'program' if spelled differently in State
    programme = state.get("programme", state.get("program", "Unknown"))
    context = state.get("retrieved_context", "NO_RETRIEVAL_NEEDED")

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student.\n"
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific details for "
            f"different programmes, highlight the one relevant to {programme}. if posible\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}"
            f"give a clear, friendly, and precise answer"
        )

    # Invoke the LLM to generate the personalized response
    response = llm.invoke(prompt)
    
    # Return the updated state. 
    # LangGraph's add_messages will append this new AI message to the chat history.
    return {"messages": [("ai",response.content.strip())]}

# step 4 - RouterFunction

def route_query(state:State):
    if state['query_type']  == 'academic':
        return "academic_rag"
    elif state['query_type']  == 'fee':
        return "fee_rag"
    else:
        return "general"

# step 5 - Graph creation 

graph = StateGraph(State)

graph.add_node("classifier",classifier_node)
graph.add_node("academic_rag",academic_rag_node)
graph.add_node("fee_rag",fee_rag_node)
graph.add_node("general",general_node)
graph.add_node("response",response_node)


# step 6 - edges 

graph.add_edge(START,"classifier")

graph.add_conditional_edges(
    "classifier",route_query
)
graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("general","response")

graph.add_edge("response",END)
app = graph.compile()


print("Welcome to Clg Assistent ")
print("1. BCA")
print("2. BBA")
print("3. B.COM")

choice = input("\nEnter 1, 2 or 3 : ")

programme_map = {
    "1" : "BCA",
    "2" : "BBA",
    "3" : "B.COM"
}

student_programme = programme_map.get(choice , "BCA")

print(f"\n Great! You set as a {student_programme} student")

while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit","quit"]:
        break
    result = app.invoke({
        "programme" : student_programme,
        "messages" : [("human",user_query)]
    })

    print(f"Assistant : {result['messages'][-1].content}")