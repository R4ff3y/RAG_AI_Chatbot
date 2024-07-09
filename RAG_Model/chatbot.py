import dotenv
import torch
from langchain.chains import LLMChain
from transformers import AutoModelForSequenceClassification
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from gpt4all import GPT4All
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
import PyPDF2
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from elasticsearch import Elasticsearch
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableSequence
from langchain_elasticsearch import ElasticsearchStore
import elasticsearch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_API_KEY"))



dotenv.load_dotenv()

review_template_str = """Your job is to provide information about the given documents.
Be as detailed as possible, but don't make up any information
that's not from the context. Also, include the chapter where you retrieved the information from.
Do not answer anything that is not related to the given context.
If you don't know an answer, say you don't know.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

output_parser = StrOutputParser()

# Instantiate the model. Callbacks support token-wise streaming
model_name = "mistralai/Mistral-7B-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
chat_model = model

def pdf_to_text(file_path):
    """Extract text from a PDF file."""
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def process_pdfs_in_folder(input_folder):
    """Process all PDFs in the specified folder."""
    all_text = ""
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_path}")
            all_text += pdf_to_text(file_path) + "\n"
    return all_text

# Specify the input folder containing PDF files
input_folder = "RAG_Model\\Drinmach"

# Process all PDFs in the input folder
loader = process_pdfs_in_folder(input_folder)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_text(loader)

# Embed and store the documents in Elasticsearch
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
es = Elasticsearch("http://localhost:9200")

# Create an index in Elasticsearch
index_name = "test-basic"

# Define the index settings and mappings
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": 1536}  # Adjust dims based on your embedding model
        }
    }
}

# Create the index
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=index_settings)
db = ElasticsearchStore.from_texts(
    docs,
    embeddings,
    es_url="http://localhost:9200",
    index_name="test-basic",
)

# Refresh the Elasticsearch index
db.client.indices.refresh(index="test-basic")

def retriever(query):
    results = db.similarity_search(query, 5)
    return results

# Function to convert string to tensor
def text_to_tensor(text, tokenizer):
    tokens = tokenizer(text, return_tensors='pt')
    return tokens.input_ids.to('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_name)

def run_chat_model(prompt):
    tokens = text_to_tensor(prompt, tokenizer)
    output = chat_model.generate(tokens,max_new_tokens= 150000)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Create a function to extract text from ChatPromptValue
def extract_text_from_prompt_value(prompt_value):
    if isinstance(prompt_value, dict):
        return prompt_value.get('context', '') + prompt_value.get('question', '')
    return str(prompt_value)

# Create a runnable sequence for the chain
review_chain = (
    RunnablePassthrough()  # Initial passthrough for context
    | review_prompt_template  # The prompt template
    | (lambda prompt: run_chat_model(extract_text_from_prompt_value(prompt)))  # Convert prompt to tensor and pass through model
    | StrOutputParser()  # String output parser
)

