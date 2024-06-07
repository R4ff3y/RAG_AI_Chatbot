# RAG_AI_Chatbot
## Preface
For our Code to run you need an [OpenAI](https://openai.com/de-DE/) Account. 
## Prequisites
Set up a Virtual Python 3.10 Environment with [VENV](https://realpython.com/python-virtual-environments-a-primer/).
Run the following commands inside the active Environment: \
` python -m pip install langchain==0.1.0 openai==1.7.2 langchain-openai==0.0.2 langchain-community==0.0.12 langchainhub==0.1.14 ` \
` python -m pip install python-dotenv `\
` python -m pip install chromadb==0.4.22 `\
`pip install gradio` \
`pip install PyPDF2` \
`pip install PyMuPDF` \

I could be missing some, check what libraries are missing and install them according to their respective documentation. \
inside the main folder create a  .env file and insert your OpenAI API Key. OPENAI_API_KEY=[YOUR KEY] \
Create a folder named input inside the `data` folder and insert the needed PDFs.
## Extracting the data
Navigate inside the data folder and run `python PDF_Extraction_Stable.py` \
When asked call the Collection "langchain".\
This will:
* Turn the PDF into text.
* Split the text into smaller chunks
* Embed the Text into a Database using OpenAI Ada embedding \
It will also:
* Extract all images of the PDF \
* Use GPT4 to summarize them \
* Embed the summary into the database using OpenAI Ada embedding \
Congratulations your data is now inside of the database.

 ## System prompt
 The System prompt is defined inside of `chatbot.py`. Feel free to adjust to achieve better results. It should include the phrases:
 * Only answer using data from the given documents
 * Do not makeup data but instead say "I don't know"
 * Include the Chapter in which you found the data \
And Some form of explanation that its Job is to assist with Questions regarding the Document.
## Query
Any user question will be sent to the database using an enhanced text query using Ada. Inside of `chatbot.py`
The line `reviews_retriever = reviews_vector_db.as_retriever(k=30)` retrieves the data from the database. \
The parameter `k` selects how many chunks from the database should be received. If `k` is too large you risk receiving inaccurate answers.
If `k` is too small, there will be insufficient data sent to the chatbot, yielding no usable result. 

## Chatbot
To run the Chatbot locally execute `python main.py`. This will start a local server instance you can access using your browser. \
To run it online change `demo.launch()` to `demo.launch("shared=True")`. The Interface is made using gradio. 
