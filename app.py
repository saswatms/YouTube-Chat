"""
YouTube Chat - Streamlit App
A simple interface to chat with YouTube videos using RAG
"""

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import re

# Page config
st.set_page_config(page_title="YouTube Chat", page_icon="🎥", layout="wide")

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stTextInput>div>div>input {
        font-size: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown('<p class="main-header">🎥 YouTube Chat</p>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: gray;">Ask questions about any YouTube video with transcripts</p>',
    unsafe_allow_html=True,
)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model selection
    model = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index=0,
        help="Choose the LLM model",
    )

    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="Controls randomness. Lower = more focused",
    )

    # Max tokens
    max_tokens = st.number_input(
        "Max Response Tokens",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Maximum length of response",
    )

    st.divider()

    st.markdown("### 📝 About")
    st.markdown(
        """
    I am [Saswat Mishra](https://github.com/saswatms).

    This app uses:
    - **Groq** for fast LLM inference
    - **LangChain** for RAG pipeline
    - **FAISS** for vector search
    - **HuggingFace** for embeddings
    """
    )

    st.markdown("### 🔗 Links")
    st.markdown("[Get Groq API Key](https://console.groq.com/)")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📹 Video Input")

    # YouTube URL input
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste any YouTube video URL",
    )

    # Extract video ID
    def extract_video_id(url):
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
            r"(?:embed\/)([0-9A-Za-z_-]{11})",
            r"^([0-9A-Za-z_-]{11})$",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    video_id = extract_video_id(youtube_url) if youtube_url else None

    # Show video preview
    if video_id:
        st.video(f"https://www.youtube.com/watch?v={video_id}")

    # Load transcript button
    if st.button("🔄 Load Transcript", type="primary", disabled=not video_id):
        if not video_id:
            st.error("⚠️ Please enter a valid YouTube URL")
        else:
            with st.spinner("Loading transcript..."):
                try:
                    # Fetch transcript
                    ytt_api = YouTubeTranscriptApi()
                    transcript_list = ytt_api.fetch(video_id, languages=["en"])
                    transcript = " ".join(chunk.text for chunk in transcript_list)

                    # Store in session state with video ID
                    st.session_state.transcript = transcript
                    st.session_state.current_video_id = video_id

                    # Clear old RAG components to rebuild with new video
                    if "chain" in st.session_state:
                        del st.session_state.chain
                    if "messages" in st.session_state:
                        st.session_state.messages = []

                    st.success(f"✅ Transcript loaded! ({len(transcript)} characters)")
                    st.rerun()

                except TranscriptsDisabled:
                    st.error("❌ This video doesn't have captions available")
                except Exception as e:
                    st.error(f"❌ Error loading transcript: {str(e)}")

with col2:
    st.subheader("💬 Chat")

    # Check if transcript is loaded
    if "transcript" in st.session_state:
        # First, check if we have API key
        if "api_key" not in st.session_state or not st.session_state.api_key:
            # Show API key input
            api_key_input = st.text_input(
                "Enter your Groq API Key to continue",
                type="password",
                placeholder="gsk_...",
                help="Get your free API key from https://console.groq.com/",
                key="api_key_input",
            )

            if api_key_input:
                st.session_state.api_key = api_key_input
                st.rerun()
            else:
                st.info("👆 Please enter your Groq API key to start chatting")
                st.markdown("[Get free API key](https://console.groq.com/)")
        else:
            # We have API key, now check if we need to build/rebuild RAG
            need_rebuild = False

            # Check if chain doesn't exist
            if "chain" not in st.session_state:
                need_rebuild = True
            # Check if model changed
            elif st.session_state.get("chain_model") != model:
                need_rebuild = True
            # Check if video changed
            elif st.session_state.get("chain_video_id") != st.session_state.get(
                "current_video_id"
            ):
                need_rebuild = True

            if need_rebuild:
                # Build RAG system
                with st.spinner("Building RAG system..."):
                    try:
                        # Split text
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, chunk_overlap=200
                        )
                        chunks = splitter.create_documents(
                            [st.session_state.transcript]
                        )

                        # Create embeddings (only once, reuse if possible)
                        if "embeddings" not in st.session_state:
                            st.session_state.embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2"
                            )
                        embeddings = st.session_state.embeddings

                        # Create vector store
                        vector_store = FAISS.from_documents(chunks, embeddings)
                        retriever = vector_store.as_retriever(
                            search_type="mmr", search_kwargs={"k": 4}
                        )

                        # Create prompt
                        prompt = PromptTemplate(
                            template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}

Answer:""",
                            input_variables=["context", "question"],
                        )

                        # Create LLM (only once per model/settings change)
                        llm_key = f"{model}_{temperature}_{max_tokens}"
                        if (
                            "llm" not in st.session_state
                            or st.session_state.get("llm_key") != llm_key
                        ):
                            st.session_state.llm = ChatGroq(
                                model=model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                api_key=st.session_state.api_key,
                            )
                            st.session_state.llm_key = llm_key

                        # Build chain
                        chain = (
                            RunnableParallel(
                                context=retriever
                                | RunnableLambda(
                                    lambda docs: "\n\n".join(
                                        d.page_content for d in docs
                                    )
                                ),
                                question=RunnablePassthrough(),
                            )
                            | prompt
                            | st.session_state.llm
                            | StrOutputParser()
                        )

                        st.session_state.chain = chain
                        st.session_state.chain_model = model
                        st.session_state.chain_video_id = (
                            st.session_state.current_video_id
                        )
                        st.success("✅ RAG system ready!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error building RAG system: {str(e)}")
                        if "api" in str(e).lower() or "auth" in str(e).lower():
                            st.error("Invalid API key. Please check and try again.")
                            if st.button("Reset API Key"):
                                del st.session_state.api_key
                                st.rerun()

            # If chain is ready, show chat interface
            if "chain" in st.session_state:
                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input
                if question := st.chat_input("Ask a question about the video..."):
                    # Add user message
                    st.session_state.messages.append(
                        {"role": "user", "content": question}
                    )
                    with st.chat_message("user"):
                        st.markdown(question)

                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                response = st.session_state.chain.invoke(question)
                                st.markdown(response)
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": response}
                                )
                            except Exception as e:
                                error_msg = f"❌ Error: {str(e)}"
                                st.error(error_msg)
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": error_msg}
                                )

                # Action buttons
                col_clear, col_reset = st.columns(2)
                with col_clear:
                    if st.button("🗑️ Clear Chat"):
                        st.session_state.messages = []
                        st.rerun()

                with col_reset:
                    if st.button("🔄 Reset App"):
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.rerun()

    else:
        st.info("👈 Enter a YouTube URL and click 'Load Transcript' to start chatting")

        # Example videos
        st.markdown("#### 📺 Try these examples:")
        example_urls = [
            "https://www.youtube.com/watch?v=d95J8yzvjbQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ]
        for url in example_urls:
            if st.button(f"Load: {url}", key=url):
                st.session_state.example_url = url
                st.rerun()

# Footer
st.divider()
st.markdown(
    """
<p style="text-align: center; color: gray; font-size: 0.8rem;">
Made with ❤️ using Streamlit, LangChain, and Groq | 
<a href="https://console.groq.com/" target="_blank">Get Free API Key</a>
</p>
""",
    unsafe_allow_html=True,
)
