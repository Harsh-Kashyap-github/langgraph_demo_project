from dotenv import load_dotenv
from typing import Annotated,Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel,Field
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class MessageClassifier(BaseModel):
    message_type:Literal["emotional","logical"]=Field(...,description=" Classify if the message requires an emotional or logical message")

class State(TypedDict):
    messages: Annotated[list,add_messages]
    message_type: str | None
    next:str | None


def classify_message(state:State):
    last_message=state["messages"][-1]
    classifying_llm=llm.with_structured_output(MessageClassifier)
    result=classifying_llm.invoke([{"role":"system","content":"""Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """},
            {"role":"user","content":last_message.content}])
    return {"message_type":result.message_type}

def router(state:State):
    if state["next"]=="emotional":
        return {"next":"emotional"}
    else:
        return {"next":"logical"}

def therapist_agent(state:State):
    last_message=state["messages"][-1]
    reply=llm.invoke([{"role":"user","content":last_message.content},{
        "role":"system","content":"""You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
    }])
    return {"messages":{"role":"assistant","content":reply.content}}
def logic_agent(state:State):
    last_message=state["messages"][-1]
    reply=llm.invoke([{"role":"user","content":last_message.content},{
        "role":"system","content":"""You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""
    }])
    return {"messages":{"role":"assistant","content":reply.content}}


graph_builder=StateGraph(State)

graph_builder.add_node("classifier",classify_message)
graph_builder.add_node("router",router)
graph_builder.add_node("therapist_agent",therapist_agent)
graph_builder.add_node("logic_agent",logic_agent)

graph_builder.add_edge(START,"classifier")
graph_builder.add_edge("classifier","router")
graph_builder.add_conditional_edges("router",lambda state:state.get("next"),{"therapist":"therapist_agent","logical":"logic_agent"})
graph_builder.add_edge("therapist_agent",END)
graph_builder.add_edge("logic_agent",END)

graph=graph_builder.compile()

def run_chatbot()->None:
    state={"messages":[],"message_type":None,"next":None}
    while True:
        user_msg=input("Enter your message:")
        if user_msg=="exit":
            print("Goodbye")
            break
        state["messages"].append({"role":"user","content":user_msg})

        state=graph.invoke(state)

        if  state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")

if __name__=="__main__":
    run_chatbot()