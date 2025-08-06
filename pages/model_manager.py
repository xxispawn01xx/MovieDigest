"""
Enhanced Model Management page with Hugging Face and Ollama integration.
"""
import streamlit as st
import os
from pathlib import Path
import json
import logging

from core.model_downloader import ModelDownloader

logger = logging.getLogger(__name__)

def show_model_manager():
    """Show the enhanced model management interface."""
    
    st.title("🤖 AI Model Manager")
    st.markdown("Download and manage AI models for video summarization")
    
    # Initialize model downloader
    if 'model_downloader' not in st.session_state:
        st.session_state.model_downloader = ModelDownloader()
    
    downloader = st.session_state.model_downloader
    
    # Tabs for different model types
    tab1, tab2, tab3 = st.tabs(["🎙️ Whisper Models", "🧠 LLM Models", "⚙️ Settings"])
    
    with tab1:
        show_whisper_models(downloader)
    
    with tab2:
        show_llm_models(downloader)
        
    with tab3:
        show_model_settings()

def show_whisper_models(downloader):
    """Show Whisper model management."""
    
    st.subheader("Speech Recognition Models")
    
    # Get available and downloaded models
    available_models = downloader.list_available_whisper_models()
    downloaded_models = downloader.list_downloaded_whisper_models()
    
    # Display current status
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Available Models", len(available_models))
    with col2:
        st.metric("Downloaded Models", len(downloaded_models))
    
    # Model recommendations
    st.markdown("### 💡 Model Recommendations")
    
    video_length = st.slider("Typical video length (minutes)", 30, 300, 90)
    quality_pref = st.selectbox("Quality preference", 
                               ["speed", "balanced", "quality"],
                               index=1)
    
    recommended = downloader.suggest_whisper_model(video_length, quality_pref)
    st.info(f"Recommended model for {video_length}min videos: **{recommended}**")
    
    # Available models table
    st.markdown("### 📋 Available Models")
    
    for model_name, description in available_models.items():
        with st.expander(f"{model_name} - {description}"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                is_downloaded = model_name in downloaded_models
                status = "✅ Downloaded" if is_downloaded else "⬇️ Not Downloaded"
                st.write(f"Status: {status}")
                
            with col2:
                if model_name == recommended:
                    st.markdown("**🎯 Recommended**")
            
            with col3:
                if not is_downloaded:
                    if st.button(f"Download {model_name}", key=f"whisper_{model_name}"):
                        with st.spinner(f"Downloading {model_name}..."):
                            success = downloader.download_whisper_model(model_name)
                            if success:
                                st.success(f"Successfully downloaded {model_name}!")
                                st.rerun()
                            else:
                                st.error(f"Failed to download {model_name}")

def show_llm_models(downloader):
    """Show LLM model management with multiple sources."""
    
    st.subheader("Language Models for Narrative Analysis")
    
    # Source selection
    source_tab1, source_tab2, source_tab3 = st.tabs(["🤗 Hugging Face", "🦙 Ollama", "📁 Local Files"])
    
    with source_tab1:
        show_huggingface_models(downloader)
    
    with source_tab2:
        show_ollama_models(downloader)
        
    with source_tab3:
        show_local_models(downloader)

def show_huggingface_models(downloader):
    """Show Hugging Face model download interface."""
    
    st.markdown("### 🤗 Hugging Face Hub")
    
    # Hugging Face Token input
    st.markdown("#### Authentication")
    
    # Check if token is stored in secrets
    hf_token = st.session_state.get('hf_token', '')
    if 'HF_TOKEN' in os.environ:
        hf_token = os.environ['HF_TOKEN']
        st.info("✅ Hugging Face token loaded from environment")
    
    token_input = st.text_input(
        "Hugging Face Token (optional for public models)", 
        value=hf_token,
        type="password",
        help="Required for private models, optional for public models"
    )
    
    if token_input != hf_token:
        st.session_state.hf_token = token_input
    
    st.markdown("#### Recommended Models")
    
    recommended_models = {
        "distilgpt2": {
            "size": "~320MB",
            "description": "Lightweight, good for basic narrative analysis",
            "type": "public"
        },
        "gpt2-medium": {
            "size": "~1.5GB", 
            "description": "Better quality, balanced performance",
            "type": "public"
        },
        "microsoft/DialoGPT-medium": {
            "size": "~1.2GB",
            "description": "Specialized for dialogue understanding",
            "type": "public"
        },
        "microsoft/DialoGPT-large": {
            "size": "~3GB",
            "description": "High quality dialogue/narrative analysis",
            "type": "public"
        }
    }
    
    for model_name, info in recommended_models.items():
        with st.expander(f"{model_name} - {info['size']}"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Type:** {info['type'].title()} model")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Check if already downloaded
                model_dir = downloader.llm_dir / model_name.replace('/', '_')
                is_downloaded = model_dir.exists()
                status = "✅ Downloaded" if is_downloaded else "⬇️ Available"
                st.write(f"Status: {status}")
            
            with col2:
                if not is_downloaded:
                    if st.button(f"Download", key=f"hf_{model_name}"):
                        download_huggingface_model(downloader, model_name, st.session_state.get('hf_token'))
    
    # Custom model input
    st.markdown("#### Custom Model")
    
    custom_model = st.text_input(
        "Model identifier",
        placeholder="e.g., microsoft/DialoGPT-small",
        help="Enter any Hugging Face model identifier"
    )
    
    if custom_model:
        col1, col2 = st.columns([2, 1])
        with col2:
            if st.button("Download Custom Model", key="custom_hf"):
                download_huggingface_model(downloader, custom_model, st.session_state.get('hf_token'))

def show_ollama_models(downloader):
    """Show Ollama model management interface."""
    
    st.markdown("### 🦙 Ollama Models")
    
    # Check Ollama installation
    try:
        import subprocess
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("✅ Ollama is installed and ready")
            
            # Popular Ollama models
            ollama_models = {
                "llama2": {
                    "size": "~3.8GB",
                    "description": "Meta's Llama 2 model, good general performance"
                },
                "mistral": {
                    "size": "~4.1GB", 
                    "description": "High-quality instruction following model"
                },
                "codellama": {
                    "size": "~3.8GB",
                    "description": "Code-focused version of Llama 2"
                },
                "llama2:7b": {
                    "size": "~3.8GB",
                    "description": "7B parameter version of Llama 2"
                },
                "mistral:7b": {
                    "size": "~4.1GB",
                    "description": "7B parameter Mistral model"
                }
            }
            
            st.markdown("#### Available Models")
            
            for model_name, info in ollama_models.items():
                with st.expander(f"{model_name} - {info['size']}"):
                    st.markdown(f"**Description:** {info['description']}")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        # Check if model exists in Ollama
                        try:
                            list_result = subprocess.run(['ollama', 'list'], 
                                                       capture_output=True, text=True)
                            is_available = model_name in list_result.stdout
                            status = "✅ Available" if is_available else "⬇️ Not Downloaded"
                            st.write(f"Status: {status}")
                        except:
                            st.write("Status: Unknown")
                    
                    with col2:
                        if st.button(f"Pull", key=f"ollama_{model_name}"):
                            download_ollama_model(downloader, model_name)
            
            # Custom Ollama model
            st.markdown("#### Custom Ollama Model")
            custom_ollama = st.text_input(
                "Model name",
                placeholder="e.g., llama2:13b",
                help="Enter any Ollama model name"
            )
            
            if custom_ollama:
                col1, col2 = st.columns([2, 1])
                with col2:
                    if st.button("Pull Model", key="custom_ollama"):
                        download_ollama_model(downloader, custom_ollama)
            
        else:
            st.warning("⚠️ Ollama not detected")
            st.markdown("**Install Ollama:**")
            st.code("curl -fsSL https://ollama.ai/install.sh | sh", language="bash")
            st.markdown("Or visit: https://ollama.ai")
            
    except FileNotFoundError:
        st.error("❌ Ollama not installed")
        st.markdown("**Install Ollama:**")
        st.markdown("1. Visit: https://ollama.ai")
        st.markdown("2. Download and install for your OS")
        st.markdown("3. Restart this app")

def show_local_models(downloader):
    """Show local model file management."""
    
    st.markdown("### 📁 Local Model Files")
    
    # Check existing local models
    local_models = []
    llm_dir = downloader.llm_dir
    
    if llm_dir.exists():
        for item in llm_dir.iterdir():
            if item.is_dir():
                # Check for model files
                has_config = (item / "config.json").exists()
                has_model = any(item.glob("*.bin")) or any(item.glob("*.safetensors"))
                
                if has_config and has_model:
                    local_models.append({
                        'name': item.name,
                        'path': str(item),
                        'size': sum(f.stat().st_size for f in item.glob("*") if f.is_file())
                    })
    
    if local_models:
        st.success(f"✅ Found {len(local_models)} local model(s)")
        
        for model in local_models:
            with st.expander(f"📦 {model['name']}"):
                st.markdown(f"**Path:** `{model['path']}`")
                st.markdown(f"**Size:** {model['size'] / (1024*1024*1024):.1f} GB")
                
                if st.button(f"Test {model['name']}", key=f"test_{model['name']}"):
                    test_local_model(model['path'])
    else:
        st.info("📂 No local models found")
    
    # Instructions for manual installation
    st.markdown("#### Manual Installation")
    st.markdown("To manually add models:")
    st.code(f"""
# Create directory structure
mkdir -p {llm_dir}/your_model_name

# Copy model files to:
{llm_dir}/your_model_name/
├── config.json
├── pytorch_model.bin (or model.safetensors)
├── tokenizer.json
└── tokenizer_config.json
    """)

def show_model_settings():
    """Show model configuration settings."""
    
    st.subheader("⚙️ Model Settings")
    
    # Environment variables section
    st.markdown("### 🔐 Environment Variables")
    
    # Hugging Face Token
    col1, col2 = st.columns([2, 1])
    with col1:
        if 'HF_TOKEN' in os.environ:
            st.success("✅ HF_TOKEN is set")
        else:
            st.warning("⚠️ HF_TOKEN not set (optional for public models)")
    
    with col2:
        if st.button("Set HF_TOKEN"):
            st.info("Add your token in the Secrets panel or .env file")
    
    # Model paths
    st.markdown("### 📁 Model Directories")
    
    if 'model_downloader' in st.session_state:
        downloader = st.session_state.model_downloader
        
        st.code(f"""
Models Directory: {downloader.models_dir}
├── whisper/     (Whisper models cache)
├── llm/         (Local LLM models) 
└── ollama/      (Ollama model references)
        """)
    
    # Storage usage
    st.markdown("### 💾 Storage Usage")
    
    if 'model_downloader' in st.session_state:
        downloader = st.session_state.model_downloader
        
        total_size = 0
        if downloader.models_dir.exists():
            total_size = sum(f.stat().st_size for f in downloader.models_dir.rglob("*") if f.is_file())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Storage", f"{total_size / (1024*1024*1024):.1f} GB")
        with col2:
            whisper_cache = Path.home() / ".cache" / "whisper"
            whisper_size = 0
            if whisper_cache.exists():
                whisper_size = sum(f.stat().st_size for f in whisper_cache.glob("*") if f.is_file())
            st.metric("Whisper Cache", f"{whisper_size / (1024*1024*1024):.1f} GB")
        with col3:
            llm_size = 0
            if downloader.llm_dir.exists():
                llm_size = sum(f.stat().st_size for f in downloader.llm_dir.rglob("*") if f.is_file())
            st.metric("LLM Models", f"{llm_size / (1024*1024*1024):.1f} GB")

def download_huggingface_model(downloader, model_name, hf_token):
    """Download a Hugging Face model with progress indication."""
    
    with st.spinner(f"Downloading {model_name} from Hugging Face..."):
        success = downloader.download_huggingface_model(model_name, hf_token)
        
        if success:
            st.success(f"✅ Successfully downloaded {model_name}!")
            st.info("Model is now available for narrative analysis")
            st.rerun()
        else:
            st.error(f"❌ Failed to download {model_name}")
            st.error("Check logs for details or verify model name and token")

def download_ollama_model(downloader, model_name):
    """Download an Ollama model with progress indication."""
    
    with st.spinner(f"Pulling {model_name} via Ollama..."):
        success = downloader.download_ollama_model(model_name)
        
        if success:
            st.success(f"✅ Successfully pulled {model_name}!")
            st.info("Model is now available via Ollama integration")
            st.rerun()
        else:
            st.error(f"❌ Failed to pull {model_name}")
            st.error("Check that Ollama is running and model name is correct")

def test_local_model(model_path):
    """Test if a local model can be loaded."""
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        with st.spinner("Testing model loading..."):
            # Try to load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            st.success("✅ Tokenizer loaded successfully")
            
            # Try to load model (just config check)
            config = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
            st.success("✅ Model config loaded successfully")
            
            st.info("Model appears to be compatible with the narrative analyzer")
            
    except Exception as e:
        st.error(f"❌ Model test failed: {str(e)}")
        st.error("This model may not be compatible")

if __name__ == "__main__":
    show_model_manager()