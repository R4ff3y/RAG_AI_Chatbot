import os
import PyPDF2
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import fitz  # PyMuPDF
from PIL import Image
import io
from openai import OpenAI
import base64
import glob
import dotenv

dotenv.load_dotenv()

# Function to convert PDF to text
def pdf_to_text(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range( len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    pdf_file.close()
    return text

# Initialize text splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

# Initialize Chroma DB client
client = chromadb.PersistentClient(path="../chroma_data")

processed_documents = set()

# Process each PDF in the ./input directory
def file_processing():
    for filename in os.listdir('./input'):
        if filename.endswith('.pdf'):
            # Convert PDF to text
            text = pdf_to_text(os.path.join('./input', filename))

            # Split text into chunks
            chunks = text_splitter.split_text(text)

            # Convert chunks to vector representations and store in Chroma DB
            documents_list = []
            embeddings_list = []
            ids_list = []

            for i, chunk in enumerate(chunks):
                if chunk not in processed_documents:
                    vector = embeddings.embed_query(chunk)
                    documents_list.append(chunk)
                    embeddings_list.append(vector)
                    ids_list.append(f"{filename}_{i}")
                    processed_documents.add(chunk)
                else: print("Duplicate!")

            if (len(ids_list) != 0):
                collection.add(
                    embeddings=embeddings_list,
                    documents=documents_list,
                    ids=ids_list
                )

def main():

    coll_name = str(input("Please enter the collection name: "))

    curr_colls = client.list_collections()

    if (any(coll.name == coll_name for coll in curr_colls)):
        client.delete_collection(name=coll_name)

    global collection 
    collection = client.create_collection(name=coll_name)
    file_processing()

    # TODO bilder löschen am ende
    files = glob.glob('data\pictures')
    for f in files:
        os.remove(f)

if __name__ == '__main__':
    main()