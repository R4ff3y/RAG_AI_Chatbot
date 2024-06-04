import os
import PyPDF2
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import fitz  # PyMuPDF
from PIL import Image
import io
import openai
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

            pdf_path = 'inputs/' + filename
            images = extract_images_from_pdf(pdf_path)

            descriptions = []
            for idx, image in enumerate(images):
                image_path = f"temp_image_{idx}.png"
                image.save(image_path)  # Save the image to disk to pass it to GPT-4
                description = get_image_description(image_path)
                descriptions.append(description)

            picture_embeddings = []
            for description in descriptions:
                embedding = get_embedding(description)
                picture_embeddings.append(embedding)

            for i, embedding in enumerate(picture_embeddings):
                metadata = {"description": descriptions[i]}
                collection.add(
                    embeddings=embedding,
                    metadatas=metadata
                )
            
            
            collection.add(
                embeddings=embeddings_list,
                documents=documents_list,
                ids=ids_list
            )


def extract_images_from_pdf(pdf_path):
    images = []
    pdf_document = fitz.open(pdf_path)
    
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    
    return images

def get_image_description(image_path):
    # Use GPT-4 with image capabilities to get a description
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides image descriptions."},
            {"role": "user", "content": "Describe the following image."},
            {"role": "user", "image": open(image_path, "rb").read()}
        ]
    )
    description = response.choices[0].message['content']
    return description

def get_embedding(text):
    return embeddings.embed_query(text)

def main():

    coll_name = str(input("Please enter the collection name: "))

    # client.delete_collection(name=coll_name)

    global collection 
    collection = client.create_collection(name=coll_name)
    file_processing()

if __name__ == '__main__':
    main()