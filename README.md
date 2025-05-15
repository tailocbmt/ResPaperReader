# Research Paper Reader

A lightweight LLM-based Research Assistant Agent that enables users to perform tasks through natural language commands.

## Authors

- Euhid Aman - M11315803
- Alexander Morinvil

## Features

- **Search internal paper databases**: Upload and search your personal collection of research papers
- **Upload new research papers**: Extract and store metadata from PDFs automatically
- **Search for recent papers**: Query public APIs like arXiv and Semantic Scholar
- **Compare papers**: Generate structured comparison reports between different papers
- **Natural language interface**: Interact with the system using everyday language

## System Architecture

The system combines:
- **RAG (Retrieval-Augmented Generation)**: For searching and retrieving relevant papers
- **ReAct (Reasoning + Acting)**: For dynamic tool invocation based on user queries
- **LLM-powered analysis**: For paper comparison, summarization, and understanding

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI (Gemini) API key
- Internet connection for external paper searches
- Raspberry Pi compatible

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/euhidaman/ResPaperReader.git
cd ResPaperReader
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all necessary packages including:
- streamlit (for the frontend interface)
- google-generativeai (for LLM capabilities)
- pdfplumber (for PDF processing)
- scikit-learn and faiss-cpu (for vector embeddings)
- arxiv and semanticscholar (for API access)

### Step 3: Get a Gemini API Key

1. Go to [Google AI Studio](https://ai.google.dev/)
2. Sign up or log in
3. Navigate to API keys section
4. Create a new API key

### Step 4: Set Up Environment Variables

You can provide your API key in one of three ways:

1. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

2. Pass it as a command line argument when running the app:
   ```bash
   python run.py --gemini-key=your_api_key_here
   ```

3. Enter it directly in the app's sidebar interface

## Running the Application

Run the application using the provided script:

```bash
python run.py
```

Or specify a custom port:

```bash
python run.py --port=8502
```

The application will start and be accessible at http://localhost:8501 (or your specified port).

## Using the Application

### Chat Interface

The chat interface lets you interact with the research assistant using natural language:

- **Search queries**: "Find papers about contrastive learning for vision models"
- **Upload requests**: "I want to upload my latest research paper"
- **Search specific sources**: "Find recent papers from ICLR about diffusion models"
- **Comparison requests**: "Compare the paper I uploaded with the second paper about diffusion models"

### Manual Navigation

Use the sidebar to navigate between different sections:

1. **Chat Assistant**: Main conversational interface
2. **Upload Papers**: Manually upload PDF research papers
3. **Search Papers**: Search papers across different sources:
   - Internal library (your uploaded papers)
   - External sources (arXiv and Semantic Scholar)
   - Conference-specific searches
4. **My Library**: View your collection of uploaded papers

## Example Workflow

1. Upload a research paper via the "Upload Papers" tab
2. Search for related papers using natural language in the chat or the "Search Papers" tab
3. Compare papers using the comparison tool at the bottom of the search results
4. Review the generated comparison report
5. Continue the conversation with follow-up questions

## Data Storage

- Papers metadata is stored in SQLite database (`data/papers.db`)
- Uploaded PDFs are saved in the `data/uploads` directory
- Vector embeddings are stored using FAISS index for semantic search capabilities

## Raspberry Pi Optimization

This application was optimized for Raspberry Pi usage:
- Lightweight vector embedding model
- SQLite instead of MySQL/PostgreSQL
- Resource-efficient components

## Troubleshooting

- **API Key Issues**: Ensure your Google Gemini API key is valid and has sufficient quota
- **PDF Extraction Problems**: Some PDFs might not be properly parsed depending on their format
- **Memory Usage**: On resource-constrained devices like Raspberry Pi, avoid processing very large PDFs or running too many operations simultaneously

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.