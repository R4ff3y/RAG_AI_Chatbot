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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableSequence
from langchain_elasticsearch import ElasticsearchStore
import elasticsearch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
login(token="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")

REVIEWS_CHROMA_PATH = "chroma_data/"

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
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    pdf_file.close()
    return text

loader = pdf_to_text("wow.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_text(loader)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
db = ElasticsearchStore.from_texts(
    docs,
    embeddings,
    es_url="http://localhost:9200",
    index_name="test-basic",
)

db.client.indices.refresh(index="test-basic")

#def retriever(query):
#    results = db.similarity_search(query, 5)
#    print(results)
#    return results
def retriever(query, threshold=0.8):
    # Retrieve initial results
    results = db.similarity_search(query, 30)
    unique_results = []
    seen_vectors = []

    for result in results:
        # Debug: Print the structure of result
        print("Result:", result)
        
        # Ensure the text field is correctly accessed
        text = result['text'] if 'text' in result else result.get('content', '')

        if not text:
            continue

        # Get the vector representation of the result
        vector = embeddings.embed_query(text)
        
        # Check similarity with seen vectors
        if seen_vectors:
            similarities = cosine_similarity([vector], seen_vectors)
            max_similarity = np.max(similarities)
        else:
            max_similarity = 0

        # If the max similarity is below the threshold, consider it as unique
        if max_similarity < threshold:
            unique_results.append(result)
            seen_vectors.append(vector)

    return unique_results


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

