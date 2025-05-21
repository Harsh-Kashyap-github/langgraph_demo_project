from dotenv import load_dotenv
from typing import Annotated,Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel,Field
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
BASE_URL="https://openrouter.ai/api/v1"

llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class State(TypedDict):
    messages: Annotated[list,add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages":[llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)

graph=graph_builder.compile()

user_input=input("Enter message:")
state=graph.invoke({"messages":[{"role":"user","content":user_input}]})

print(state["messages"][-1].content)