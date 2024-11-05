from openai import OpenAI
import streamlit as st
import time
import io
import os
from denser_retriever.keyword import (
    ElasticKeywordSearch,
    create_elasticsearch_client,
)
from denser_retriever.retriever import DenserRetriever
import fitz
import json
import logging
import anthropic
import argparse

logger = logging.getLogger(__name__)

# Define available models
MODEL_OPTIONS = {
    "GPT-4": "gpt-4o",
    "Claude 3.5": "claude-3-5-sonnet-20241022"
}
context_window = 128000
# Get API keys from environment variables with optional default values
openai_api_key = os.getenv('OPENAI_API_KEY')
claude_api_key = os.getenv('CLAUDE_API_KEY')

# Check if API keys are set
if not openai_api_key and not claude_api_key:
    raise ValueError("Neither OPENAI_API_KEY nor CLAUDE_API_KEY environment variables is set")

openai_client = OpenAI(api_key=openai_api_key)
claude_client = anthropic.Client(api_key=claude_api_key)
history_turns = 5



prompt_default = "### Instructions:\n" \
                 "You are a professional AI assistant. The following context consists of an ordered list of sources. " \
                 "If you can find answers from the context, use the context to provide a response. " \
                 "You must cite passages in square brackets [X] where X is the passage number (the ranking order of provided passages)." \
                 "If you cannot find the answer from the sources, use your knowledge to come up a reasonable answer. " \
                 "If the query asks to summarize the file or uploaded file, provide a summarization based on the provided sources. " \
                 "If the conversation involves casual talk or greetings, rely on your knowledge for an appropriate response. "


def get_annotation_pages(annotations_str):
    """Get all unique page numbers from annotations."""
    try:
        annotations = json.loads(annotations_str)
        if annotations and isinstance(annotations, list):
            return sorted(set(ann.get('page', 0) for ann in annotations))
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass
    return []


def pdf_viewer(file_path, page_num=0, annotations=None):
    """Displays a single page of a PDF in Streamlit with optional annotations."""
    doc = fitz.open(file_path)
    page = doc[page_num]

    # Draw annotations (highlights) if any exist for the page
    for ann in (annotations or []):
        if ann['page'] == page_num:
            rect = fitz.Rect(ann['x'], ann['y'], ann['x'] + ann['width'], ann['y'] + ann['height'])
            # Highlight the area with light blue color
            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=(1, 1, 0))  # Light blue
            highlight.update()

    # Render page as an image
    # Increase zoom factor for better resolution
    zoom = 2  # Adjust this value to change the resolution
    mat = fitz.Matrix(zoom, zoom)
    image = page.get_pixmap(matrix=mat)

    # Use st.columns to create full width container
    col1 = st.columns(1)[0]
    with col1:
        st.image(image.tobytes(), use_column_width=True)


def render_pdf():
    """Render PDF with annotations and handle page navigation."""
    try:
        if st.session_state.current_annotations:
            annotations = []
            try:
                annotations = json.loads(st.session_state.current_annotations)
                if st.session_state.clicked:
                    st.session_state.current_page = annotations[0].get('page', 0)
                    st.session_state.clicked = False
            except json.JSONDecodeError:
                st.error("Invalid annotation format")

        pdf_path = st.session_state.current_pdf
        if pdf_path:
            file = open(pdf_path, "rb")
            doc = fitz.open(stream=io.BytesIO(file.read()), filetype="pdf")
            total_pages = doc.page_count

            # Navigation controls
            nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])

            # Previous Page button
            with nav_col1:
                if st.button("Previous Page", key="prev_btn",
                             disabled=(st.session_state.current_page <= 0)):
                    st.session_state.current_page -= 1
                    st.rerun()

            # Page number display
            with nav_col2:
                st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")

            # Next Page button
            with nav_col3:
                if st.button("Next Page", key="next_btn",
                             disabled=(st.session_state.current_page >= total_pages - 1)):
                    st.session_state.current_page += 1
                    st.rerun()

            # Display the PDF viewer with the current page and annotations
            pdf_viewer(
                pdf_path,
                page_num=st.session_state.current_page,
                annotations=annotations
            )

    except Exception as e:
        st.error(f"Error rendering PDF: {str(e)}")


def stream_response(selected_model, messages, passages):
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        if selected_model == "gpt-4o":
            print("Using OpenAI GPT-4 model")
            messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
            for response in openai_client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    stream=True,
                    top_p=0,
                    temperature=0.0
            ):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        else:
            print("Using Claude 3.5 model")
            with claude_client.messages.stream(
                    max_tokens=1024,
                    messages=messages,
                    model="claude-3-5-sonnet-20241022",
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)

            message_placeholder.markdown(full_response, unsafe_allow_html=True)

    # Update session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.passages = passages

    # Rerun to show the updated UI with passages
    st.rerun()


def main(args):
    # Set page configuration to use wide mode
    st.set_page_config(layout="wide")

    global retriever
    retriever = DenserRetriever(
        index_name=args.index_name,
        keyword_search=ElasticKeywordSearch(
            top_k=100,
            es_connection=create_elasticsearch_client(url="http://localhost:9200",
                                                      username="elastic",
                                                      password="",
                                                      ),
            drop_old=False,
            analysis="default"  # default or ik
        ),
        vector_db=None,
        reranker=None,
        embeddings=None,
        gradient_boost=None,
        search_fields=["annotations:keyword"],
    )

    # Initialize session states
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Claude 3.5"  # Default model
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    if 'current_pdf' not in st.session_state:
        st.session_state.current_pdf = None
    if 'current_annotations' not in st.session_state:
        st.session_state.current_annotations = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "passages" not in st.session_state:
        st.session_state.passages = []

    # Create two columns for main content and PDF viewer
    main_col, pdf_col = st.columns([1, 1])

    with main_col:
        # Create a header row with title and model selector
        st.title("Denser Chat Demo")
        selected_model_name = st.selectbox(
            "Select Model",
            options=list(MODEL_OPTIONS.keys()),
            key="model_selector",
            index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model)
        )
        st.session_state.selected_model = selected_model_name

        st.caption(
            "Try question \"What is in-batch negative sampling ?\" or \"what parts have stop pins?\"")
        st.divider()

        if len(st.session_state.messages) > 1:
            with st.chat_message(st.session_state.messages[-2]["role"]):
                st.markdown(st.session_state.messages[-2]["content"], unsafe_allow_html=True)

        # Display passages and add annotation buttons
        if st.session_state.passages:  # Show passages if they exist
            num_passages = len(st.session_state.passages)
            buttons_per_row = 5

            # Calculate number of rows needed
            num_rows = (num_passages + buttons_per_row - 1) // buttons_per_row

            for row in range(num_rows):
                # Create columns for this row
                start_idx = row * buttons_per_row
                end_idx = min(start_idx + buttons_per_row, num_passages)
                num_buttons_this_row = end_idx - start_idx

                # Create columns for this row
                cols = st.columns(num_buttons_this_row)

                # Add buttons to columns
                for col_idx, passage_idx in enumerate(range(start_idx, end_idx)):
                    passage = st.session_state.passages[passage_idx]
                    annotations = passage[0].metadata.get('annotations', '[]')
                    pages = get_annotation_pages(annotations)
                    page_str = f"Source {passage_idx + 1}" if pages else "No annotations"
                    # print(f"Passage {passage_idx}: {passage[0].page_content}")

                    with cols[col_idx]:
                        if st.button(page_str, key=f"btn_page_{passage_idx}"):
                            st.session_state.current_pdf = passage[0].metadata.get('source', None)
                            st.session_state.current_annotations = annotations
                            st.session_state.clicked = True
                            st.rerun()

        if len(st.session_state.messages) > 0:
            with st.chat_message(st.session_state.messages[-1]["role"]):
                st.markdown(st.session_state.messages[-1]["content"], unsafe_allow_html=True)

        # Handle user input
        query = st.chat_input("Please input your question")
        if query:
            with st.chat_message("user"):
                st.markdown(query)

            start_time = time.time()
            passages = retriever.retrieve(query, 5, {})
            retrieve_time_sec = time.time() - start_time
            st.write(f"Retrieve time: {retrieve_time_sec:.3f} sec.")

            # Process chat completion
            prompt = prompt_default + f"### Query:\n{query}\n"
            if len(passages) > 0:
                prompt += f"\n### Context:\n"
                for i, passage in enumerate(passages):
                    prompt += f"#### Passage {i+1}:\n{passage[0].page_content}\n"

            if args.language == "en":
                context_limit = 4 * context_window
            else:
                context_limit = context_window
            prompt = prompt[:context_limit] + "### Response:"

            # Prepare messages for chat completion
            messages = st.session_state.messages[-history_turns * 2:]
            messages.append({"role": "user", "content": prompt})

            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": query})

            stream_response(MODEL_OPTIONS[selected_model_name], messages, passages)

    # Render PDF viewer in the second column
    with pdf_col:
        render_pdf()

def parse_args():
    parser = argparse.ArgumentParser(description='Denser Chat Demo')
    parser.add_argument('--index_name', type=str, default=None,
                      help='Name of the Elasticsearch index to use')
    parser.add_argument('--language', type=str, default='en',
                      help='Language setting for context window (en or ch, default: en)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)