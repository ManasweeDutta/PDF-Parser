import os
import PyPDF2
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Streamlit App
st.title("CartMapper")

# Generate a sample PDF with inventory data
def generate_sample_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(72, 750, "Inventory Report")
    c.drawString(72, 730, "Date: 2023-10-01")
    c.drawString(72, 710, "Item ID: 001, Name: Laptop, Quantity: 10, Price: $1200")
    c.drawString(72, 690, "Item ID: 002, Name: Mouse, Quantity: 50, Price: $20")
    c.drawString(72, 670, "Item ID: 003, Name: Keyboard, Quantity: 30, Price: $50")
    c.drawString(72, 650, "Item ID: 004, Name: Monitor, Quantity: 15, Price: $300")
    c.drawString(72, 630, "Item ID: 005, Name: Laptop Bag, Quantity: 25, Price: $40")
    c.drawString(72, 610, "Item ID: 006, Name: USB-C Hub, Quantity: 40, Price: $30")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Provide a sample PDF for testing
if st.button("Generate Sample Inventory PDF"):
    sample_pdf = generate_sample_pdf()
    st.download_button(
        label="Download Sample Inventory PDF",
        data=sample_pdf,
        file_name="sample_inventory.pdf",
        mime="application/pdf"
    )

if uploaded_file:
    try:
        # Process uploaded PDF
        data = []

        with uploaded_file as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                data.append({
                    'page_number': page_num + 1,
                    'page_content': page_text
                })

        documents = [Document(page_content=page['page_content']) for page in data]

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Set up API key
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Create vector database
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="huggingface-groq-rag"
        )

        # Initialize Groq LLM (using Mixtral-8x7b or Llama 2-70b)
        llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768",  # Use "llama2-70b-4096" for Llama 2
            groq_api_key=GROQ_API_KEY
        )

        # Prompt for query reformulation
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI assistant generating alternative query perspectives.
            Generate 5 different versions of the given question to improve document retrieval:
            Original question: {question}"""
        )

        # MultiQueryRetriever setup
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(),
            llm,
            prompt=QUERY_PROMPT
        )

        # Define prompt template for answering questions and suggesting complementary goods
        template = """Answer the question based ONLY on the following context:
        {context}

        Question: {question}

        Additionally, suggest 3 complementary goods that pair well with the item mentioned in the question.
        For example, if the question is about a laptop, suggest items like a mouse, keyboard, or laptop bag.
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Define the chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # User input for query
        query = st.text_input("Enter your question:", "How may I help you?")

        if st.button("Get Answer"):
            # Invoke the chain
            result = chain.invoke(query)
            st.write("### Answer:")
            st.write(result)

    except Exception as e:
        st.error(f"An error occurred: {e}")