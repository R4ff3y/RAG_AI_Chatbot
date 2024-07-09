# RAG_AI_Chatbot
## Preface
For our Code to run you need an [OpenAI](https://openai.com/de-DE/) Account. 
For our Code to run you need an [Huggingface](https://huggingface.co/) Account
## Prequisites
Set up a Virtual Python 3.10 Environment with [VENV](https://realpython.com/python-virtual-environments-a-primer/).



inside the main folder create a  .env file and insert your OpenAI API Key. OPENAI_API_KEY=[YOUR KEY] and Huggingface API Key HUGGINGFACE_API_KEY=[YOUR KEY]\
Create a folder named Drinmach inside the `RAG_Model` folder and insert the needed PDFs.
## Extracting the data
Works like:
* Turn the PDF into text.
* Split the text into smaller chunks
* Embed the Text into a Database using OpenAI Ada embedding \
Congratulations your data is now inside of the database.

 ## System prompt
 The System prompt is defined inside of `chatbot.py`. Feel free to adjust to achieve better results. It should include the phrases:
 * Only answer using data from the given documents
 * Do not makeup data but instead say "I don't know"
 * Include the Chapter in which you found the data \
And Some form of explanation that its Job is to assist with Questions regarding the Document.
## Query
OUTDATED Any user question will be sent to the database using an enhanced text query using Ada. Inside of `chatbot.py`
The line `reviews_retriever = reviews_vector_db.as_retriever(k=30)` retrieves the data from the database. \
The parameter `k` selects how many chunks from the database should be received. If `k` is too large you risk receiving inaccurate answers.
If `k` is too small, there will be insufficient data sent to the chatbot, yielding no usable result. 

## Chatbot
To run the Chatbot locally execute `python main.py`. This will start a local server instance you can access using your browser. \
To run it online change `demo.launch()` to `demo.launch("shared=True")`. The Interface is made using gradio. \
For most local models you need a lot of computationalpower. \
If you want to swap out the models, you can use any model that works with the huggingface langchain api. Inform accordingly [here](https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/)
