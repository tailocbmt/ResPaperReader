from models.research_assistant import ResearchAssistant
import streamlit as st
import os
import sys
import logging
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Page configuration
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'assistant' not in st.session_state:
    # Get API key from secrets or environment
    api_key = os.environ.get("GEMINI_API_KEY", "")

    # Initialize the research assistant
    st.session_state.assistant = ResearchAssistant(gemini_api_key=api_key)
    st.session_state.chat_history = []
    st.session_state.current_papers = []
    st.session_state.api_key_set = bool(api_key)

# Helper functions


def display_paper(paper, index=None):
    """Display a paper in the UI."""
    with st.container():
        col1, col2 = st.columns([9, 1])

        title = paper.get('title', 'Unknown Title')

        with col1:
            st.markdown(f"### {index+1 if index is not None else ''}) {title}")

        with col2:
            if st.button("Analyze", key=f"analyze_{hash(title)}"):
                if paper.get('id'):
                    with st.spinner("Analyzing paper..."):
                        analysis = st.session_state.assistant.analyze_paper(
                            paper['id'])
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"## Analysis of '{title}'\n\n{analysis.get('analysis', 'Analysis failed')}"
                    })

        # Show authors if available
        authors = paper.get('authors', [])
        if authors:
            if isinstance(authors, list):
                st.markdown(f"**Authors:** {', '.join(authors)}")
            else:
                st.markdown(f"**Authors:** {authors}")

        # Show abstract with a "Read more" expander if it's long
        abstract = paper.get('abstract', '')
        if len(abstract) > 200:
            st.markdown(f"**Abstract:** {abstract[:200]}...")
            with st.expander("Read more"):
                st.markdown(abstract)
        else:
            st.markdown(f"**Abstract:** {abstract}")

        # Show source info
        source = paper.get('source', 'Unknown')
        if source == 'arxiv':
            st.markdown(
                f"**Source:** arXiv | [Paper Link]({paper.get('url', '#')}) | [PDF]({paper.get('pdf_url', '#')})")
        elif source == 'semantic_scholar':
            st.markdown(
                f"**Source:** Semantic Scholar | [Paper Link]({paper.get('url', '#')})")
        else:
            st.markdown(f"**Source:** {source}")

        st.markdown("---")


def display_comparison_selector():
    """Display UI for selecting papers to compare."""
    st.subheader("Compare Papers")

    # Get papers from memory and database
    papers = st.session_state.current_papers

    if not papers or len(papers) < 2:
        st.warning("You need at least 2 papers in your results to compare.")
        return

    col1, col2 = st.columns(2)

    with col1:
        paper1_idx = st.selectbox(
            "Select first paper:",
            options=range(len(papers)),
            format_func=lambda i: papers[i].get('title', f"Paper {i+1}"),
            key="compare_paper1"
        )

    with col2:
        paper2_idx = st.selectbox(
            "Select second paper:",
            options=range(len(papers)),
            format_func=lambda i: papers[i].get('title', f"Paper {i+1}"),
            key="compare_paper2"
        )

    if st.button("Generate Comparison"):
        if paper1_idx == paper2_idx:
            st.error("Please select two different papers to compare.")
            return

        with st.spinner("Generating comparison report..."):
            # Use the actual paper IDs if available, otherwise use session indices
            paper1_id = papers[paper1_idx].get('id', str(paper1_idx))
            paper2_id = papers[paper2_idx].get('id', str(paper2_idx))

            comparison = st.session_state.assistant.generate_paper_comparison(
                paper_id_1=paper1_id,
                paper_id_2=paper2_id
            )

            # Add to chat history
            title1 = papers[paper1_idx].get('title', f"Paper {paper1_idx+1}")
            title2 = papers[paper2_idx].get('title', f"Paper {paper2_idx+1}")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"## Comparison: '{title1}' vs '{title2}'\n\n{comparison}"
            })


# Sidebar
with st.sidebar:
    st.title("Research Paper Assistant")

    # API Key Input
    with st.expander("API Settings", expanded=not st.session_state.api_key_set):
        api_key_input = st.text_input(
            "Enter Gemini API Key:",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password"
        )

        if st.button("Save API Key"):
            os.environ["GEMINI_API_KEY"] = api_key_input
            st.session_state.assistant = ResearchAssistant(
                gemini_api_key=api_key_input)
            st.session_state.api_key_set = bool(api_key_input)
            st.success("API key saved!")

    # Navigation
    st.header("Navigation")
    page = st.radio("Go to:", ["Chat Assistant",
                    "Upload Papers", "Search Papers", "My Library"])

# Main area
if page == "Chat Assistant":
    st.header("Research Paper Chat Assistant")

    # Display chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Assistant:** {content}")

    # Input for new messages
    user_input = st.chat_input("Ask a question about research papers...")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input})

        # Process query
        with st.spinner("Processing your query..."):
            response = st.session_state.assistant.process_natural_language_query(
                user_input)

        # Handle different response types
        if response["action"] == "upload_prompt":
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response["message"]})
            # Auto-switch to upload page
            st.switch_page(page)

        elif response["action"] == "tool_results":
            # Display base message
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response["message"]})

            # Process and display results
            for result in response.get("results", []):
                tool_name = result["tool"]
                result_data = result["result"]

                if tool_name in ["internal_search", "web_search", "conference_search"]:
                    # Update current papers for comparison
                    st.session_state.current_papers = result_data

                    # Show results
                    result_message = f"## Search Results\nFound {len(result_data)} papers:\n\n"
                    for i, paper in enumerate(result_data[:5]):  # Show top 5
                        title = paper.get('title', 'Unknown Title')
                        result_message += f"{i+1}. {title}\n"

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result_message})

                elif tool_name == "compare_papers":
                    # Result is already a formatted comparison
                    if isinstance(result_data, str):
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": result_data})
        else:
            # Simple response
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response["message"]})

elif page == "Upload Papers":
    st.header("Upload Research Papers")

    uploaded_file = st.file_uploader("Upload a PDF research paper", type="pdf")

    if uploaded_file:
        with st.spinner("Processing paper..."):
            result = st.session_state.assistant.upload_paper(uploaded_file)

        if result["success"]:
            st.success(f"Successfully uploaded: {result['title']}")

            # Display paper details
            st.subheader("Paper Details")
            st.markdown(f"**Title:** {result['title']}")
            st.markdown(
                f"**Authors:** {', '.join(result['authors']) if result['authors'] else 'Unknown'}")
            st.markdown(f"**Abstract:** {result['abstract']}")

            # Add to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Paper '{result['title']}' has been uploaded and processed."
            })
        else:
            st.error(
                f"Failed to upload paper: {result.get('message', 'Unknown error')}")

elif page == "Search Papers":
    st.header("Search for Research Papers")

    # Tabs for different search types
    tab1, tab2, tab3 = st.tabs(
        ["Internal Search", "Web Search", "Conference Search"])

    with tab1:
        st.subheader("Search Internal Library")
        query = st.text_input("Enter search terms:", key="internal_search")

        if st.button("Search Internal Library"):
            with st.spinner("Searching internal database..."):
                results = st.session_state.assistant.search_internal_papers(
                    query)

            st.session_state.current_papers = results

            st.subheader(f"Search Results ({len(results)})")
            if results:
                for i, paper in enumerate(results):
                    display_paper(paper, i)
            else:
                st.info("No papers found. Try uploading papers first.")

    with tab2:
        st.subheader("Search External Sources")
        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input("Enter search terms:", key="web_search")

        with col2:
            source = st.selectbox(
                "Source:",
                options=[None, "arxiv", "semantic_scholar"],
                format_func=lambda x: "Both" if x is None else x.capitalize()
            )

        if st.button("Search External Sources"):
            with st.spinner("Searching external sources..."):
                results = st.session_state.assistant.search_web_papers(
                    query, source)

            st.session_state.current_papers = results

            st.subheader(f"Search Results ({len(results)})")
            if results:
                for i, paper in enumerate(results):
                    display_paper(paper, i)
            else:
                st.info("No papers found. Try different search terms.")

    with tab3:
        st.subheader("Search by Conference")
        col1, col2 = st.columns(2)

        with col1:
            conference = st.text_input(
                "Conference name (e.g., ICLR, NeurIPS, ACL):")

        with col2:
            year = st.text_input("Year (optional):")

        if st.button("Search Conference Papers"):
            if not conference:
                st.warning("Please enter a conference name.")
            else:
                with st.spinner(f"Searching for papers from {conference}..."):
                    results = st.session_state.assistant.search_conference_papers(
                        conference, year)

                st.session_state.current_papers = results

                st.subheader(f"Search Results ({len(results)})")
                if results:
                    for i, paper in enumerate(results):
                        display_paper(paper, i)
                else:
                    st.info("No papers found. Try a different conference or year.")

    # Paper comparison tool (appears at the bottom of search page)
    st.markdown("---")
    display_comparison_selector()

elif page == "My Library":
    st.header("My Paper Library")

    with st.spinner("Loading your library..."):
        papers = st.session_state.assistant.db.get_all_papers()

    st.session_state.current_papers = papers

    st.subheader(f"Your Papers ({len(papers)})")
    if papers:
        for i, paper in enumerate(papers):
            display_paper(paper, i)
    else:
        st.info("No papers in your library yet. Try uploading or searching for papers.")

    # Paper comparison tool
    st.markdown("---")
    display_comparison_selector()
