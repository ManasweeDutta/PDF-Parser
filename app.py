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

# Streamlit App
st.title("CartMapper")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
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

    # Initialize LLM
    llm = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        groq_api_key=os.environ['GROQ_API_KEY']
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

    # Define prompt template for answering questions
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
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

