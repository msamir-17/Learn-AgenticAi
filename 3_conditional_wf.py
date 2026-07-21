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