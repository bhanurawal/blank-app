###https://github.com/pkycode/RAGAPP/blob/main/RAG_PDF_App.py

from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import SentenceTransformer

def split_text_batch(text_batch):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len, add_start_index=True)
    return text_splitter.split_text(text_batch)

def generate_embeddings_batch(chunks_batch, model):
    return model.encode(chunks_batch, convert_to_tensor=True)

def main():
    st.title("File Upload and Q&A App")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        binary_data = uploaded_file.getvalue()
        text = read_pdf_pypdf2(uploaded_file)
        
        # Split text in batches
        batch_size = 1000
        text_batches = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]
        
        with ProcessPoolExecutor() as executor:
            chunks_batches = list(executor.map(split_text_batch, text_batches))
        
        # Flatten the list of chunks batches
        chunks = [chunk for batch in chunks_batches for chunk in batch]
        
        # Generate embeddings in batches
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings_batches = list(executor.map(generate_embeddings_batch, chunks_batches, [model]*len(chunks_batches)))
        
        # Flatten the list of embeddings batches
        embeddings = [embedding for batch in embeddings_batches for embedding in batch]
        
        st.write(embeddings)

if __name__ == "__main__":
    main()
