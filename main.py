from langchain_intro.chatbot import review_chain

# question = str(input("Please enter your question: "))

print(review_chain.invoke(question))