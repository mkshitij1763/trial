# Chat with Multiple PDFs

## Project Overview

This project is a Streamlit application that allows users to upload multiple PDF documents and interact with them using natural language questions. The application uses natural language processing (NLP) techniques to extract text from PDFs, create embeddings for document retrieval, and generate responses using a pre-trained T5 model. The system leverages FAISS for efficient similarity search and Sentence-BERT for embedding generation.

## Installation Instructions

Follow these steps to set up the environment and run the project:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/chat-with-multiple-pdfs.git
    cd chat-with-multiple-pdfs
    ```

2. **Set up a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the T5 model:**
    Run the following script to download and save the T5 model locally.
    ```python
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
    model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")

    tokenizer.save_pretrained("MBZUAI/LaMini-T5-738M")
    model.save_pretrained("MBZUAI/LaMini-T5-738M")
    ```

5. **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Upload PDF Documents:**
   - In the sidebar, upload one or more PDF files by clicking on the "Upload your PDFs here and click on 'Process'" button.
   - Click the "Process" button to extract and process text from the uploaded PDFs.

2. **Ask Questions:**
   - Once the PDFs are processed, type your question in the text input box at the top of the page.
   - Press Enter to submit your question.
   - The system will retrieve relevant text from the documents and generate a response, which will be displayed in the chat interface.

### Example

1. **Upload PDFs:**
   - Upload a PDF document containing information about a specific topic (e.g., a research paper).

2. **Ask a Question:**
   - Type a question such as "What are the main findings of the study?"
   - The system retrieves relevant sections from the document and generates a response summarizing the main findings.

## Dependencies

- `streamlit`: Web application framework for creating the interactive UI.
- `PyPDF2`: Library for reading PDF files.
- `langchain`: Utility for splitting text into manageable chunks.
- `sentence-transformers`: Library for creating embeddings using Sentence-BERT.
- `faiss`: Library for efficient similarity search.
- `transformers`: Library for loading and using pre-trained NLP models.

### requirements.txt
```text
streamlit==1.6.0
PyPDF2==2.12.1
langchain==0.4.0
sentence-transformers==2.2.2
faiss-cpu==1.7.2
transformers==4.19.2
numpy==1.21.4
```

Ensure you have the correct versions of the libraries by installing them using the provided `requirements.txt` file.

## Detailed Report

### 1. Introduction

#### Problem Statement
In today's digital age, vast amounts of information are stored in PDF documents, including research papers, manuals, reports, and e-books. Extracting meaningful information from these documents can be challenging, especially when dealing with multiple PDFs. The objective of this project is to develop a tool that allows users to interact with multiple PDF documents through natural language queries, providing quick and accurate responses by leveraging natural language processing (NLP) techniques.

#### Objectives
- Develop a system to extract and process text from multiple PDF documents.
- Implement a method to retrieve relevant document sections based on user queries.
- Generate coherent and contextually appropriate responses using a pre-trained language model.
- Provide an intuitive user interface for uploading PDFs and asking questions.

### 2. Approach

#### Methodology

##### 2.1 PDF Text Extraction
The first step involves extracting text from PDF documents. This is achieved using the `PyPDF2` library, which allows us to read and extract text from each page of the PDF.

```python
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
```

##### 2.2 Text Chunking
The extracted text is then split into manageable chunks using the `CharacterTextSplitter` from the `langchain` library. This ensures that the text chunks are of appropriate length for processing.

```python
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
```

##### 2.3 Embedding Generation
To retrieve relevant documents based on user queries, we generate embeddings for the text chunks using the Sentence-BERT model (`all-MiniLM-L6-v2`) from the `sentence-transformers` library.

```python
def get_vectorstore(text_chunks):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(text_chunks, convert_to_tensor=False)
    embeddings = np.array(embeddings)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return FAISSRetriever(index=index, texts=text_chunks, embeddings=embeddings)
```

##### 2.4 Document Retrieval
A FAISS index is created to store the embeddings, enabling efficient similarity search. The `FAISSRetriever` class is used to fetch the most relevant documents based on the query embedding.

```python
class FAISSRetriever:
    def __init__(self, index, texts, embeddings):
        self.index = index
        self.texts = texts
        self.embeddings = embeddings

    def get_relevant_documents(self, query, k=5):
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k=k)
        return [self.texts[i] for i in I[0]]
```

##### 2.5 Response Generation
We use a pre-trained T5 model (`LaMini-T5-738M`) from the `transformers` library to generate responses based on the retrieved document text. The model is fine-tuned for conversational tasks.

```python
def get_conversation_model():
    local_model_path = "MBZUAI/LaMini-T5-738M"
    tokenizer = T5Tokenizer.from_pretrained(local_model_path)
    model = T5ForConditionalGeneration.from_pretrained(local_model_path)
    return tokenizer, model

def generate_response(input_text, tokenizer, model, max_length=100):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    response = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(response[0], skip_special_tokens=True)
```

##### 2.6 User Interface
The Streamlit library is used to create an interactive web interface. Users can upload PDF files, input queries, and view the responses generated by the system.

```python
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "file_details" not in st.session_state:
        st.session_state.file_details = []

    st.header("Chat with multiple PDFs :books:")
    query = st.text_input("Ask a question about your documents:")

    if query and "conversation" in st.session_state:
        vectorstore = st.session_state.conversation
        relevant_documents = vectorstore.get_relevant_documents(query)
        
        if relevant_documents:
            combined_text = "\n".join(relevant_documents)
            if "tokenizer" not in st.session_state or "model" not in st.session_state:
                tokenizer, model = get_conversation_model()
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
            else:
                tokenizer = st.session_state.tokenizer
                model = st.session_state.model
            
            response = generate_response(combined_text, tokenizer, model)
            st.write(user_template.replace("{{MSG}}", query), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = vectorstore
                        st.session_state.file_details = [{"name": pdf.name, "size": pdf.size, "data": pdf} for pdf in pdf_docs

]
                        st.success("Processing completed. Now ask a question!")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                    
            if st.session_state.file_details:
                st.markdown("<div class='file-review'><h3>File Preview</h3>", unsafe_allow_html=True)
                for file_detail in st.session_state.file_details:
                    st.markdown(f"<p><strong>Name:</strong> {file_detail['name']}<br><strong>Size:</strong> {file_detail['size']} bytes</p>", unsafe_allow_html=True)
                    pdf_data = file_detail['data'].getvalue()
                    pdf_display = render_pdf(pdf_data)
                    st.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
```

### 3. Failed Approaches

#### Initial Embedding Methods
Initially, we experimented with other embedding models such as `distilbert-base-nli-mean-tokens` from the `sentence-transformers` library. However, these models resulted in less accurate document retrieval, possibly due to their larger size and different training objectives.

#### Non-chunked Text Processing
We attempted to process entire documents without splitting them into chunks. This approach failed due to memory constraints and reduced retrieval accuracy. The documents were too large for effective embedding, and the responses lacked specificity.

#### Direct Query to T5 Model
Another approach was to directly input the user query and the entire document text to the T5 model. This method was inefficient and often resulted in the model generating irrelevant or overly verbose responses.

### 4. Results

#### Metrics and Performance
- **Text Extraction**: Successfully extracted text from various PDF documents.
- **Embedding Generation**: Created embeddings for text chunks with an average size of 1000 characters.
- **Document Retrieval**: Retrieved relevant documents with high accuracy for different queries.
- **Response Generation**: Generated coherent and contextually appropriate responses.

#### Visualizations
1. **Text Chunking Visualization**
   ![Text Chunking](images/text_chunking.png)
   *Figure 1: Visualization of text chunking process.*

2. **Embedding Space Visualization**
   ![Embedding Space](images/embedding_space.png)
   *Figure 2: PCA visualization of document embeddings.*

#### Example Query and Response
- **Query**: "What are the main findings of the study?"
- **Response**: "The study found that the proposed method significantly improves performance on the benchmark dataset, achieving a new state-of-the-art accuracy."

### 5. Discussion

#### Analysis of Results
The system effectively handles multiple PDF documents, extracting and processing text for interactive querying. The combination of Sentence-BERT for embeddings and the T5 model for response generation proved to be effective. The chunking approach ensured that text was processed efficiently, allowing for accurate and relevant document retrieval.

#### Insights Gained
- **Embedding Models**: The choice of embedding model significantly impacts the accuracy of document retrieval.
- **Chunking Strategy**: Proper text chunking is crucial for efficient processing and accurate response generation.
- **Model Fine-tuning**: Fine-tuning pre-trained models on domain-specific data can further enhance performance.

### 6. Conclusion

This project demonstrates a successful implementation of an interactive tool for querying multiple PDF documents using NLP techniques. By leveraging Sentence-BERT for embeddings and a pre-trained T5 model for response generation, the system provides accurate and contextually appropriate responses. Future improvements could include fine-tuning the models on specific domains and enhancing the user interface for better user experience.

### 7. References

- **PyPDF2**: [PyPDF2 Documentation](https://pypdf2.readthedocs.io/)
- **Sentence-Transformers**: [Sentence-Transformers Documentation](https://www.sbert.net/)
- **FAISS**: [FAISS Documentation](https://github.com/facebookresearch/faiss)
- **Transformers**: [Transformers Documentation](https://huggingface.co/transformers/)
- **Streamlit**: [Streamlit Documentation](https://docs.streamlit.io/)
