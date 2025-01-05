import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def split_text(documents: list[Document]):
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
  chunks = text_splitter.split_documents(documents)
  print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

  # Print example of page content and metadata for a chunk
  document = chunks[0]
  print(document.page_content)
  print(document.metadata)

  return chunks # Return the list of split text chunks

def main():
    st.title("File Upload and Q&A App")
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Process the uploaded file
        st.write("File uploaded successfully!")
        # Add your RAG model processing code here
        binary_data = uploaded_file.getvalue()
        # pdf_viewer(input=binary_data, width=700)
        chunks = split_text(uploaded_file)
        
        question = st.text_input("Ask a question about the file")
        
        if st.button("Submit Question", type="primary"):
            if question:
                # Add your RAG model question-answering code here
                answer = "This is where the answer will be displayed."
                st.write(answer)

if __name__ == "__main__":
    main()
