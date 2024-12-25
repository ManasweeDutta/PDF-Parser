import os
import PyPDF2
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq




os.chdir("D:\Pycharm_projects")


pdf_path = "D:\\Pycharm_projects\\mono.pdf"
data = []

with open(pdf_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)


    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        page_text = page.extract_text()

        data.append({
            'page_number': page_num + 1,
            'page_content': page_text
        })


documents = [Document(page_content=page['page_content']) for page in data]


text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


os.environ['GROQ_API_KEY'] = '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'


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="huggingface-groq-rag"
)


llm = ChatGroq(
    temperature=0,
    model_name="mixtral-8x7b-32768",
    groq_api_key=os.environ['GROQ_API_KEY']
)
from langchain_core.prompts import PromptTemplate

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant generating alternative query perspectives.
    Generate 5 different versions of the given question to improve document retrieval:
    Original question: {question}"""
)


retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)


template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


result = chain.invoke("what is a monopoly?")
print(result)
