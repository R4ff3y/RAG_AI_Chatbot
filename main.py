from RAG_Model.chatbot import review_chain
import gradio as gr

# question = str(input("Please enter your question: "))

def ask_chatbot(question, history):
    return review_chain.invoke(question)

demo = gr.ChatInterface(fn=ask_chatbot, title="Hydac GPT", 
                        theme='soft')

demo.launch()