import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import PyPDF2
from sentence_transformers import SentenceTransformer, util
import torch
import os

def read_pdf_pypdf2(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
  
def split_text(textPdf):
  """
  Split the text content of the given list of Document objects into smaller chunks.
  Args:
    documents (list[Document]): List of Document objects containing text content to split.
  Returns:
    list[Document]: List of Document objects representing the split text chunks.
  """
  # Initialize text splitter with specified parameters
  text_splitter = RecursiveCharacterTextSplitter(
     chunk_size=500, # Size of each chunk in characters
     chunk_overlap=100, # Overlap between consecutive chunks
     length_function=len, # Function to compute the length of the text
     add_start_index=True, # Flag to add start index to each chunk
   )

  #  text_splitter = SemanticChunker(lc_embed_model)

  # Split documents into smaller chunks using text splitter
  chunks = text_splitter.split_text(textPdf)
  print(f"Split documents into {len(chunks)} chunks.")

  return chunks # Return the list of split text chunks

def generate_embedd(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ## df = pd.DataFrame(columns=['text', 'embedding'])
    
    # Generate embeddings for the data
    embeddings = model.encode(chunks, convert_to_tensor=True)
    ## df = pd.concat([df, pd.DataFrame({'text': [text], 'embedding': [embedding]})], ignore_index=True)

    return embeddings,model

def main():

    os.environ["api_key"] == st.secrets["API_KEY"]
    
    st.title("File Upload and Q&A App")
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Process the uploaded file
        st.write("File uploaded successfully!")
        # Add your RAG model processing code here
        binary_data = uploaded_file.getvalue()
        #### pdf_viewer(input=binary_data, width=700)
        chunks = split_text(read_pdf_pypdf2(uploaded_file))
        
        question = st.text_input("Ask a question about the file")
        
        if st.button("Submit Question", type="primary"):
            if question:
                # Add your RAG model question-answering code here
                #Create embedding for pdf
                embeddings , model = generate_embedd(chunks)
                
                # Query Embeddings
                query_embedding = model.encode(question, convert_to_tensor=True)
                
                # Compute cosine similarities
                cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
                
                # Find the top 3 most similar sentences
                top_results = torch.topk(cosine_scores, k=3)
                
                # Print results
                ### answer = "This is where the answer will be displayed."
                for score, idx in zip(top_results[0], top_results[1]):
                    #print(f"Score: {score.item():.4f}\nText: {data[idx]}\n")
                    st.write(f"Score: {score.item():.4f}\nText: {chunks[idx]}\n")

if __name__ == "__main__":
    main()
