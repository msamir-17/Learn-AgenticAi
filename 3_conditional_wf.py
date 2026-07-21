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