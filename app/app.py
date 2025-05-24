import base64
import uuid
import streamlit as st
import os
import sys
import logging
from models.enums import COMPARE_PAPERS, CONFERENCE_SEARCH, INTERNAL_SEARCH, WEB_SEARCH, LLMSource, NavigationType, WebSearchSource
from assistant.research_assistant import ResearchAssistant

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Page configuration
st.set_page_config(
    page_title="chatGPaperT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'assistant' not in st.session_state:
    # Get API key from secrets or environment
    api_key = os.environ.get("GEMINI_API_KEY", "")

    # Initialize the research assistant
    st.session_state.llm_name = LLMSource.GEMINI.value
    st.session_state.assistant = ResearchAssistant(
        gemini_api_key=api_key,
        llm_name=st.session_state.llm_name
    )
    st.session_state.chat_history = []
    st.session_state.current_papers = []
    st.session_state.api_key_set = bool(api_key)

# Helper functions


def display_paper(paper, index=None, allow_delete=False):
    """Display a paper in the UI."""
    with st.container():
        col1, col2, col3 = st.columns([8, 1, 1])

        title = paper.get('title', 'Unknown Title')

        with col1:
            st.markdown(f"### {index+1 if index is not None else ''}) {title}")

        with col2:
            if st.button("Analyze", key=f"analyze_{uuid.uuid4()}"):
                if paper.get('id'):
                    with st.spinner("Analyzing paper..."):
                        analysis = st.session_state.assistant.analyze_paper(
                            paper['id']
                        )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"## Analysis of '{title}'\n\n{analysis.get('analysis', 'Analysis failed')}"
                        }
                    )

        with col3:
            if allow_delete and paper.get('id'):
                if st.button("Delete", key=f"delete_{uuid.uuid4()}", type="primary", help="Delete this paper permanently"):
                    with st.spinner("Deleting paper..."):
                        result = st.session_state.assistant.delete_paper(
                            paper['id'])

                    if result["success"]:
                        st.success(f"Paper '{title}' deleted successfully")
                        # Add a rerun call to refresh the page
                        st.rerun()
                    else:
                        st.error(
                            f"Failed to delete paper: {result['message']}")

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

        with st.spinner("Comparing paper..."):
            # Use the actual paper IDs if available, otherwise use session indices
            paper1_id = papers[paper1_idx].get('id', str(paper1_idx))
            paper2_id = papers[paper2_idx].get('id', str(paper2_idx))

            comparison = st.session_state.assistant.compare_papers(
                paper_1=papers[paper1_idx],
                paper_2=papers[paper2_idx]
            )

            # Add to chat history
            title1 = papers[paper1_idx].get('title', f"Paper {paper1_idx+1}")
            title2 = papers[paper2_idx].get('title', f"Paper {paper2_idx+1}")

            if comparison:
                with st.container(key="comparison_detail", border=True):
                    st.subheader("Comparison Details")
                    st.markdown(
                        f"Comparison: '{title1}' vs '{title2}'\n\n{comparison}")

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"Comparison: '{title1}' vs '{title2}'\n\n{comparison}"
                }
            )


# Sidebar
with st.sidebar:
    st.title("chatGPaperT")

    # Choose LLM model
    with st.container(key="llm_setting", border=True):
        st.subheader("LLM Settings")
        llm_name_input = st.selectbox(
            label="Model:",
            options=[model.value for model in LLMSource],
            index=0,
            placeholder=LLMSource.GEMINI.value,
        )
        st.session_state.llm_name = llm_name_input
        st.session_state.assistant = ResearchAssistant(
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            llm_name=st.session_state.llm_name
        )

    # Navigation
    with st.container(key="navigation", border=True):
        st.subheader("Navigation")
        page = st.selectbox(
            label="Go to page:",
            index=0,
            options=[type.value for type in NavigationType]
        )

# Chat Assistant
if page == NavigationType.CHAT_ASSISTANT.value:
    st.header("Chat with Assistant")

    # Display chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            prefix_str = "You" if role == "user" else "Assistant"

            st.markdown(f"**{prefix_str}:** {content}")

    # Input for new messages
    user_input = st.chat_input("Ask something", key="chat assistant")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        # Process query
        with st.spinner("Processing query..."):
            response = st.session_state.assistant.process_normal_query(
                user_input
            )

        # Handle different response types
        if response["action"] == "upload_prompt":
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response["message"]
                }
            )
            # Auto-switch to upload page
            st.switch_page(page)

        elif response["action"] == "tool_results":
            # Display base message
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response["message"]
                }
            )

            # Process and display results
            for result in response.get("results", []):
                tool_name = result["tool"]
                result_data = result["result"]

                if tool_name in [INTERNAL_SEARCH, WEB_SEARCH, CONFERENCE_SEARCH]:
                    # Update current papers for comparison
                    st.session_state.current_papers = result_data

                    # Show results
                    result_message = f"## Search Results\nFound {len(result_data)} papers:\n\n"
                    # Show only top 5
                    for i, paper in enumerate(result_data[:5]):
                        title = paper.get('title', 'Unknown Title')
                        result_message += f"{i+1}. {title}\n"

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result_message
                        }
                    )
                elif tool_name == COMPARE_PAPERS:
                    # Result is already a formatted comparison. Display the comparison
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result_data
                        }
                    )
        else:
            # Simple response
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response["message"]
                }
            )

elif page == NavigationType.UPLOAD_PAPERS.value:
    st.header("Upload Research Papers")

    uploaded_file = st.file_uploader(
        "Upload research paper (PDF only)",
        type="pdf"
    )

    if uploaded_file:
        with st.spinner("Processing paper..."):
            try:
                # Get file info for debugging
                file_info = f"File: {uploaded_file.name}, Size: {uploaded_file.size} bytes, Type: {uploaded_file.type}"
                logging.info(f"Processing uploaded file: {file_info}")

                # Ensure upload directory exists
                os.makedirs(os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), "data", "uploads"), exist_ok=True)

                # Process the paper
                result = st.session_state.assistant.upload_paper(
                    uploaded_file
                )

                if result["success"]:
                    st.success(f"Successfully uploaded: {result['title']}")

                    # Display paper details
                    with st.container(key="paper_detail", border=True):
                        st.subheader("Paper Details")
                        st.markdown(f"**Title:** {result['title']}")
                        st.markdown(
                            f"**Authors:** {', '.join(result['authors']) if result['authors'] else 'Unknown'}")
                        st.markdown(f"**Abstract:** {result['abstract']}")

                    # Add to chat history
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"Paper '{result['title']}' has been uploaded and processed."
                        }
                    )
                else:
                    error_msg = result.get('message', 'Unknown error')
                    st.error(f"Failed to upload paper: {error_msg}")
                    logging.error(f"Upload failed: {error_msg}")
            except Exception as e:
                st.error(f"An error occurred during upload: {str(e)}")
                logging.exception("Unexpected error during paper upload:")

elif page == NavigationType.SEARCH_PAPERS.value:
    st.header("Search for Research Papers")

    # Tabs for different search types
    tab1, tab2, tab3 = st.tabs(
        ["Internal Search", "Web Search", "Conference Search"])

    with tab1:
        st.subheader("Internal Library")
        query = st.text_input("Enter search terms:", key="internal_search")

        if st.button("Search", key='internal_button'):
            with st.spinner("Searching internal database..."):
                results = st.session_state.assistant.search_internal(
                    query)

            st.session_state.current_papers = results

            st.subheader(f"Search Results ({len(results)})")
            if results:
                for i, paper in enumerate(results):
                    display_paper(paper, i)
            else:
                st.info("No papers found. Try uploading papers first.")

    with tab2:
        st.subheader("Search External Source")
        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input("Enter search terms:", key="web_search")

        with col2:
            source = st.selectbox(
                "Source:",
                options=[
                    "Both",
                    WebSearchSource.ARXIV.value,
                    WebSearchSource.SEMANTIC_SCHOLAR.value
                ],
                format_func=lambda x: x.capitalize()
            )

        if st.button("Search", key='external_button'):
            with st.spinner("Searching external sources..."):
                results = st.session_state.assistant.search_web(
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

        if st.button("Search", key='conference_button'):
            if not conference:
                st.warning("Please enter a conference name.")
            else:
                with st.spinner(f"Searching for papers from {conference}..."):
                    results = st.session_state.assistant.search_conference(
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

elif page == NavigationType.CHAT_WITH_PAPERS.value:
    st.header("Ask about Papers")

    # Get all papers from the database
    papers = st.session_state.assistant.db.get_papers()

    if not papers:
        st.info("Your library is empty. Please upload some papers first.")
    else:
        # Paper selector
        selected_paper = st.selectbox(
            "Select a paper to investigate:",
            options=papers,
            format_func=lambda p: p.get('title', 'Unknown Title')
        )

        if selected_paper:
            st.write("### Selected Paper")
            display_paper(selected_paper, allow_delete=True)

            # Chat interface
            st.write("### Chat")

            # Initialize paper-specific chat history in session state if not exists
            paper_chat_key = f"paper_chat_{selected_paper['id']}"
            if paper_chat_key not in st.session_state:
                st.session_state[paper_chat_key] = []

            # Display paper-specific chat history
            for message in st.session_state[paper_chat_key]:
                role = message["role"]
                content = message["content"]

                with st.chat_message(role):
                    prefix_str = "You" if role == "user" else "Assistant"
                    st.markdown(f"**{prefix_str}:** {content}")

                    if role == "assistant":
                        # If the message has sources, display them in an expander
                        if "sources" in message:
                            with st.expander("View sources from paper"):
                                for source in message["sources"]:
                                    st.markdown(f"```\n{source['text']}\n```")
                                    if "metadata" in source:
                                        st.markdown(
                                            f"*Page {source['metadata'].get('page', 'unknown')}*")

            # Chat input
            question = st.chat_input(
                f"Ask questions about '{selected_paper['title']}'...", key="chat paper")
            logging.error(question)
            if question:
                # Add user question to history
                st.session_state[paper_chat_key].append(
                    {
                        "role": "user",
                        "content": question
                    }
                )
                # Get response using RAG
                with st.spinner("Generating response..."):
                    response = st.session_state.assistant.chat_with_paper(
                        selected_paper['id'],
                        question
                    )

                if "error" in response:
                    st.error(
                        response["error"]
                    )
                else:
                    # Add assistant response with sources to history
                    st.session_state[paper_chat_key].append(
                        {
                            "role": "assistant",
                            "content": response["response"],
                            "sources": response.get("sources", [])
                        }
                    )

                # Rerun to update the chat display
                st.rerun()
