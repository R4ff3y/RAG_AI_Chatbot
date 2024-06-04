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

AI_Client = OpenAI()
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

            pdf_path = 'input/' + filename
            images = extract_images_from_pdf(pdf_path)

            descriptions = []
            for idx, image in enumerate(images):
                image_path = f"pictures/temp_image_{idx}.png"
                image.save(image_path)  # Save the image to disk to pass it to GPT-4
                description = get_image_description(image_path)
                descriptions.append(description)

            Pic_embeddings = []
            Pic_ids_list = []
            for description in descriptions:
                embedding = get_embedding(description)
                Pic_embeddings.append(embedding)

            for i, embedding in enumerate(Pic_embeddings):
                # TODO schlaue id wählen
                Pic_Id = f"{filename}_{i}"
                metadata = {"description": descriptions[i]}
                collection.add(
                    embeddings=embedding,
                    metadatas=metadata,
                    ids=Pic_Id
                )

            if (len(ids_list) != 0):
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

    # Read the image file as bytes
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Encode the image data as base64
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    # Use GPT-4 with image capabilities to get a description
#    response = AI_Client.chat.completions.create(
#        model="gpt-4o",
#        messages=[
#            {"role": "system", "content": "You are a helpful assistant that provides image descriptions."},
#            {"role": "user", "content": "Describe the following image."},
#            {"role": "user", "content": "Image description", "image": image_base64}
#        ]
#    )

    response = AI_Client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }     
                }
                ]
            }
        ]
    )

    print(response)
    description = response.choices[0].message.content
    return description

def get_embedding(text):
    return embeddings.embed_query(text)


def main():

    coll_name = str(input("Please enter the collection name: "))

    curr_colls = client.list_collections()

    if (any(coll.name == coll_name for coll in curr_colls)):
        client.delete_collection(name=coll_name)

    global collection 
    collection = client.create_collection(name=coll_name)
    file_processing()

    # TODO bilder löschen am ende

if __name__ == '__main__':
    main()