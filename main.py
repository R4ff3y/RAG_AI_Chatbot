from RAG_Model.chatbot import review_chain
from RAG_Model.chatbot import retriever

import gradio as gr

def ask_chatbot(question, history):
    
    context = retriever(question)
    input_data = {"context": context, "question": question}
    return review_chain.invoke(input_data)

demo = gr.ChatInterface(fn=ask_chatbot, title="HIVE", theme='soft')

demo.launch()