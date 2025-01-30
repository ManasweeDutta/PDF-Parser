import os
import PyPDF2
import streamlit as st
from typing import List
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Dict

class ProductRecommendation(BaseModel):
    answer: str = Field(description="The direct answer to the user's question")
    complementary_products: Dict[str, str] = Field(description="Dictionary of complementary products with brief explanations")
    reasoning: str = Field(description="Brief explanation of why these products complement each other")

class DocumentProcessor:
    @staticmethod
    def extract_pdf_text(file) -> List[dict]:
        try:
            data = []
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    data.append({
                        'page_number': page_num + 1,
                        'page_content': page_text
                    })
            return data
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return []

class RAGPipeline:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embeddings = None
        self.vector_db = None
        self.llm = None
        self.chain = None
        self.recommendation_chain = None

    def initialize_components(self, documents: List[Document]):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="huggingface-groq-rag"
        )

        self.llm = ChatGroq(
            temperature=0.2,
            model_name="mixtral-8x7b-32768",
            groq_api_key=self.api_key,
            max_tokens=1024
        )

        recommendation_template = """
        You are a retail AI assistant. Use the context to answer the question and identify complementary products.

        Context:
        {context}

        Question: {question}

        Provide:
        1. A direct answer to the question as a plain text string.
        2. A list of complementary products from the catalog.
        3. Brief explanations of why these products go together.

        Format the response as a JSON object with fields:
        - answer: "Direct response as a string"
        - complementary_products: Dictionary of complementary products with explanations
        - reasoning: Explanation of why these products complement each other
        """

        recommendation_prompt = ChatPromptTemplate.from_template(recommendation_template)
        parser = PydanticOutputParser(pydantic_object=ProductRecommendation)

        retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

        self.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | recommendation_prompt
            | self.llm
            | parser
        )

def main():
    st.set_page_config(
        page_title="CartMapper",
        page_icon="🛒",
        layout="wide"
    )
    st.title("🛒 CartMapper")
    st.subheader("Product Information & Recommendations")

    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None

    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Please set the GROQ_API_KEY in your environment or Streamlit secrets")
        return

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        with st.spinner("Processing product catalog..."):
            data = DocumentProcessor.extract_pdf_text(uploaded_file)
            if not data:
                st.error("No text could be extracted from the PDF. Try another document.")
                return

            documents = [Document(page_content=page['page_content']) for page in data]

            if st.session_state.rag_pipeline is None:
                st.session_state.rag_pipeline = RAGPipeline(api_key)
                st.session_state.rag_pipeline.initialize_components(documents)
                st.success("Product catalog processed successfully!")

        st.markdown("### Ask About Products")
        query = st.text_input("Enter your question:", placeholder="Ask about a product...")

        if st.button("Get Answer & Recommendations"):
            if query.strip():
                try:
                    with st.spinner("Generating recommendations..."):
                        result = st.session_state.rag_pipeline.chain.invoke(query)

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown("### Answer")
                            st.write(result.answer)

                        with col2:
                            st.markdown("### Recommended Together")
                            for product, explanation in result.complementary_products.items():
                                st.markdown(f"- **{product}**: {explanation}")

                            st.markdown("### Why These Go Well Together")
                            st.write(result.reasoning)
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
            else:
                st.warning("Please enter a question about a product.")

if __name__ == "__main__":
    main()