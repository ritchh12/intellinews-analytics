import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain_openai import ChatOpenAI
import unstructured

load_dotenv() 

st.set_page_config(
    page_title="IntelliNews Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #F24236;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        color: #2E86AB;
        font-weight: bold;
    }
    .status-message {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
    }
    .info {
        background-color: #d1ecf1;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 IntelliNews Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced News Research & Analysis Platform</p>', unsafe_allow_html=True)

st.sidebar.markdown('<h2 class="sidebar-header">📰 Article Sources</h2>', unsafe_allow_html=True)
st.sidebar.markdown("Enter up to 3 news article URLs for comprehensive analysis")

st.sidebar.markdown("Enter up to 3 news article URLs for comprehensive analysis")

urls = []
for i in range(3):
    url = st.sidebar.text_input(
        f"🔗 Article URL {i+1}", 
        placeholder=f"Enter news article URL {i+1}...",
        help="Paste the full URL of a news article"
    )
    urls.append(url)

st.sidebar.markdown("---")
process_url_clicked = st.sidebar.button(
    "🚀 Analyze Articles", 
    type="primary",
    help="Click to process and analyze the provided URLs"
)

VECTOR_STORE_PATH = "faiss_store_openai.pkl"
main_content_area = st.empty()

@st.cache_resource
def initialize_llm():
    """Initialize and cache the language model for efficient usage."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found in environment variables. Please check your .env file.")
        st.stop()
    
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile"
    )

llm = initialize_llm()

llm = initialize_llm()

if process_url_clicked:
    valid_urls = [url for url in urls if url.strip()]
    
    if not valid_urls:
        st.error("⚠️ Please provide at least one valid URL to proceed.")
    else:
        progress_container = st.container()
        
        with progress_container:
            st.success(f"🎯 Processing {len(valid_urls)} article(s)...")
            
            with st.spinner("📥 Loading article content..."):
                loader = UnstructuredURLLoader(urls=valid_urls)
                st.info("✅ Article data successfully loaded")
                article_data = loader.load()
            
            with st.spinner("📝 Processing and chunking text..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                st.info("✅ Text processing completed")
                processed_documents = text_splitter.split_documents(article_data)
            
            with st.spinner("🧠 Creating knowledge embeddings..."):
                model_name = "all-MiniLM-L6-v2"
                embeddings = HuggingFaceEmbeddings(model_name=model_name)
                vector_store = FAISS.from_documents(processed_documents, embeddings)
                st.info("✅ Knowledge base successfully created")
                time.sleep(1)

            with open(VECTOR_STORE_PATH, "wb") as f:
                pickle.dump(vector_store, f)
            
            st.success("🎉 Analysis complete! You can now ask questions about the articles.")

            st.success("🎉 Analysis complete! You can now ask questions about the articles.")

st.markdown("---")
st.markdown("### 🤔 Ask Your Questions")
st.markdown("Enter your question about the analyzed articles and get instant insights.")

query = st.text_input(
    "💭 Your Question:", 
    placeholder="What are the key insights from these articles?",
    help="Ask any question about the content of the analyzed articles"
)

if query:
    if os.path.exists(VECTOR_STORE_PATH):
        with st.spinner("🔍 Analyzing and generating response..."):
            with open(VECTOR_STORE_PATH, "rb") as f:
                loaded_vector_store = pickle.load(f)
                retrieval_chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, 
                    retriever=loaded_vector_store.as_retriever()
                )
                analysis_result = retrieval_chain({"question": query}, return_only_outputs=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 💡 Analysis Result")
                st.markdown(f"**Question:** {query}")
                st.write(analysis_result["answer"])
            
            with col2:
                sources = analysis_result.get("sources", "")
                if sources:
                    st.markdown("### 📚 Source References")
                    source_list = sources.split("\n")
                    for idx, source in enumerate(source_list, 1):
                        if source.strip():
                            st.markdown(f"**{idx}.** {source}")
                else:
                    st.info("No specific sources identified for this response.")
    else:
        st.warning("⚠️ Please process some articles first before asking questions.")
        st.info("Use the sidebar to add article URLs and click 'Analyze Articles' to get started.")

        st.warning("⚠️ Please process some articles first before asking questions.")
        st.info("Use the sidebar to add article URLs and click 'Analyze Articles' to get started.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>IntelliNews Analytics</strong> - Powered by Advanced AI Technology</p>
    <p>🔬 Analyze • 🧠 Understand • 📊 Insights</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ How to Use")
st.sidebar.markdown("""
1. **Add URLs**: Enter up to 3 news article URLs
2. **Analyze**: Click 'Analyze Articles' to process content
3. **Ask**: Type your questions about the articles
4. **Get Insights**: Receive AI-powered analysis with sources
""")

st.sidebar.markdown("### 🔧 Features")
st.sidebar.markdown("""
- ✅ Multi-article analysis
- ✅ Source attribution
- ✅ Advanced AI processing
- ✅ Real-time insights
""")

if os.path.exists(VECTOR_STORE_PATH):
    st.sidebar.success("📊 Knowledge base loaded and ready!")
else:
    st.sidebar.info("🔄 Ready to analyze articles")




