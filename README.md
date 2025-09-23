# 📊 [IntelliNews Analytics (click to preview)](https://ritchh12-intellinews-analytics-news-research-tool-wvt4f9.streamlit.app/)

> **Advanced News Research & Analysis Platform**

A powerful AI-driven tool for analyzing multiple news articles simultaneously and extracting meaningful insights through natural language queries.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-v0.0.340+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

- **Multi-Article Analysis** - Process up to 3 news articles simultaneously
- **AI-Powered Insights** - Advanced language model for intelligent responses
- **Source Attribution** - Clear references to original article sources
- **Real-time Processing** - Instant analysis and response generation
- **Professional UI** - Clean, intuitive interface with progress tracking
- **Vector Search** - Efficient semantic search through processed content

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: LangChain, OpenAI GPT, HuggingFace Transformers
- **Vector Database**: FAISS
- **Text Processing**: Unstructured, RecursiveCharacterTextSplitter
- **Embeddings**: all-MiniLM-L6-v2

## 📋 Requirements

- Python 3.8 or higher
- Internet connection for article processing
- Groq API key for language model access

## 🔧 Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd news-research-tool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   - Update the API key in `news_research_tool.py`
   - Replace the existing Groq API key with your own

4. **Run the application**
   ```bash
   streamlit run news_research_tool.py
   ```

## 🎯 Usage

### Step 1: Add Article URLs
- Enter up to 3 news article URLs in the sidebar
- Ensure URLs are valid and accessible

### Step 2: Process Articles
- Click "🚀 Analyze Articles" to begin processing
- Wait for the analysis to complete (progress indicators will guide you)

### Step 3: Ask Questions
- Enter your questions in the main query box
- Get AI-powered insights with source references
- Ask follow-up questions for deeper analysis

## 💡 Example Queries

- *"What are the main topics covered in these articles?"*
- *"What are the key differences between the sources?"*
- *"Summarize the most important points from all articles"*
- *"What trends or patterns can you identify?"*
- *"Are there any conflicting viewpoints presented?"*

## 📁 Project Structure

```
news-research-tool/
│
├── news_research_tool.py      # Main application file
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
└── faiss_store_openai.pkl    # Generated vector store (after first use)
```

## 🔒 API Configuration

The application uses Groq API for language model access. To configure:

1. Sign up for a Groq API account
2. Get your API key
3. Replace the API key in the `initialize_llm()` function:

```python
return ChatOpenAI(
    api_key="your-groq-api-key-here",
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile"
)
```

## 🎨 UI Features

- **Responsive Design** - Optimized for various screen sizes
- **Progress Tracking** - Real-time processing status
- **Professional Styling** - Clean, modern interface
- **Interactive Elements** - Tooltips and help text
- **Source Attribution** - Clear reference tracking

## 🚧 Troubleshooting

### Common Issues

1. **Installation Problems**
   - Ensure Python 3.8+ is installed
   - Use a virtual environment to avoid conflicts
   - Install dependencies one by one if batch install fails

2. **API Errors**
   - Verify your Groq API key is valid
   - Check internet connection
   - Ensure API quota is not exceeded

3. **Article Processing Issues**
   - Verify URLs are accessible
   - Check if articles are behind paywalls
   - Try with different news sources

## 🔄 Future Enhancements

- [ ] Multi-language support
- [ ] Export functionality (PDF, Excel)
- [ ] Sentiment analysis visualization
- [ ] Real-time news feed integration
- [ ] Advanced filtering options
- [ ] Collaborative features
- [ ] Mobile app version

## 📊 Performance

- **Processing Speed**: ~30-60 seconds for 3 articles
- **Memory Usage**: ~200-500MB depending on article length
- **Supported Formats**: Web articles, blog posts, news sites

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Feature enhancements
- Documentation improvements
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Powered by [LangChain](https://langchain.com/) for AI orchestration
- Uses [FAISS](https://faiss.ai/) for efficient vector search
- Groq API for fast language model inference

---

**Made with ❤️ for better news analysis and research**

*For support or questions, please open an issue in the repository.*
