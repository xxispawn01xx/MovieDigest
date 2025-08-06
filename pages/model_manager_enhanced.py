"""
Enhanced Model Management page with one-click LLM downloads.
"""
import streamlit as st
from pathlib import Path
import subprocess
import sys
import json
import os
import shutil
from core.model_downloader import ModelDownloader

def show_model_manager():
    """Display the enhanced model management interface with one-click downloads."""
    
    st.header("🧠 Model Management Center")
    
    # Initialize model downloader
    downloader = ModelDownloader()
    
    # Quick Stats Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        installed_models = len([f for f in Path("models").rglob("*") if f.is_file()]) if Path("models").exists() else 0
        available_storage = shutil.disk_usage(".")[2] / (1024**3)  # GB
        
        with col1:
            st.metric("Model Files", installed_models)
        
        with col2:
            st.metric("Available Storage", f"{available_storage:.1f} GB")
        
        with col3:
            try:
                import torch
                gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
            except ImportError:
                gpu_status = "CPU"
            st.metric("Processing Mode", gpu_status)
        
        with col4:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    st.metric("GPU Memory", f"{gpu_memory:.1f} GB")
                else:
                    import psutil
                    ram = psutil.virtual_memory().total / (1024**3)
                    st.metric("Total RAM", f"{ram:.1f} GB")
            except ImportError:
                st.metric("Total RAM", "Unknown")
    except Exception as e:
        st.warning("Could not retrieve system information")
    
    st.divider()
    
    # Recommended LLM Models Section with One-Click Download
    st.subheader("🎯 Recommended LLM Models - One-Click Install")
    
    recommended_models = [
        {
            "name": "Microsoft DialoGPT-medium",
            "repo": "microsoft/DialoGPT-medium", 
            "size": "1.2 GB",
            "description": "🎬 Best for movie analysis - specializes in dialogue and character interactions",
            "use_case": "Dialogue-heavy movies, character studies, drama",
            "speed": "Fast",
            "quality": "High",
            "recommended": True,
            "command": "huggingface-cli download microsoft/DialoGPT-medium --local-dir models/local_llm"
        },
        {
            "name": "GPT2-medium",
            "repo": "gpt2-medium",
            "size": "1.5 GB", 
            "description": "🎯 General purpose - excellent for action movies and documentaries",
            "use_case": "Action movies, documentaries, varied content",
            "speed": "Medium",
            "quality": "High",
            "recommended": False,
            "command": "huggingface-cli download gpt2-medium --local-dir models/local_llm"
        },
        {
            "name": "DistilGPT2 (Lightweight)",
            "repo": "distilgpt2",
            "size": "320 MB",
            "description": "⚡ Ultra-fast - perfect for testing and batch processing",
            "use_case": "Testing, batch processing, quick analysis",
            "speed": "Very Fast",
            "quality": "Good", 
            "recommended": False,
            "command": "huggingface-cli download distilgpt2 --local-dir models/local_llm"
        },
        {
            "name": "DialoGPT-large (Premium)",
            "repo": "microsoft/DialoGPT-large", 
            "size": "3.0 GB",
            "description": "🏆 Maximum quality - best results for complex narratives",
            "use_case": "Complex plots, ensemble casts, premium analysis",
            "speed": "Slower",
            "quality": "Excellent",
            "recommended": False,
            "command": "huggingface-cli download microsoft/DialoGPT-large --local-dir models/local_llm"
        }
    ]
    
    for model in recommended_models:
        # Check if model is installed
        model_path = Path("models/local_llm")
        is_installed = model_path.exists() and any(model_path.iterdir()) if model_path.exists() else False
        
        # Header with status badge
        header_text = f"{'⭐ ' if model['recommended'] else ''}{model['name']} - {model['size']}"
        if is_installed:
            header_text += " ✅ INSTALLED"
        
        with st.expander(header_text):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**{model['description']}**")
                st.write(f"🎯 **Best for:** {model['use_case']}")
                
                # Performance indicators
                perf_col1, perf_col2 = st.columns(2)
                with perf_col1:
                    st.write(f"⚡ **Speed:** {model['speed']}")
                with perf_col2:
                    st.write(f"🎨 **Quality:** {model['quality']}")
                
                # Show command for reference
                with st.expander("📋 Manual Installation Command"):
                    st.code(model['command'], language='bash')
                    st.write("You can run this command in your terminal if you prefer manual installation.")
            
            with col2:
                if is_installed:
                    st.success("✅ Installed")
                    
                    # Model info if available
                    if model_path.exists():
                        try:
                            size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024*1024)
                            st.write(f"📁 Size: {size_mb:.0f} MB")
                        except:
                            pass
                    
                    if st.button(f"🗑️ Remove", key=f"remove_{model['repo']}", use_container_width=True):
                        with st.spinner("Removing model..."):
                            try:
                                if model_path.exists():
                                    shutil.rmtree(model_path)
                                st.success("Model removed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to remove model: {e}")
                else:
                    # One-click download button
                    button_text = "🚀 One-Click Install" if model['recommended'] else "⬇️ Download"
                    button_type = "primary" if model['recommended'] else "secondary"
                    
                    if st.button(button_text, key=f"install_{model['repo']}", 
                               type=button_type, use_container_width=True):
                        
                        # Show download progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.write("📥 Starting download...")
                        progress_bar.progress(10)
                        
                        try:
                            # Use huggingface-cli to download
                            cmd = ["huggingface-cli", "download", model['repo'], "--local-dir", "models/local_llm"]
                            
                            status_text.write("📥 Downloading model files...")
                            progress_bar.progress(30)
                            
                            # Create directory if it doesn't exist
                            Path("models/local_llm").mkdir(parents=True, exist_ok=True)
                            
                            result = subprocess.run(
                                cmd,
                                capture_output=True,
                                text=True,
                                timeout=1800  # 30 minute timeout
                            )
                            
                            progress_bar.progress(90)
                            
                            if result.returncode == 0:
                                progress_bar.progress(100)
                                status_text.write("✅ Download complete!")
                                st.balloons()
                                st.success(f"🎉 {model['name']} installed successfully!")
                                
                                # Update session state to indicate model is ready
                                st.session_state.llm_model_ready = True
                                st.session_state.selected_llm_model = model['repo']
                                
                                st.rerun()
                            else:
                                status_text.write("❌ Download failed")
                                st.error(f"Failed to download {model['name']}. Error: {result.stderr}")
                                
                        except subprocess.TimeoutExpired:
                            status_text.write("❌ Download timeout")
                            st.error(f"Download of {model['name']} timed out. Please try again.")
                        except FileNotFoundError:
                            status_text.write("❌ huggingface-cli not found")
                            st.error("huggingface-cli not found. Please install it with: pip install huggingface_hub[cli]")
                        except Exception as e:
                            status_text.write("❌ Download error")
                            st.error(f"Error downloading {model['name']}: {str(e)}")
    
    st.divider()
    
    # Whisper Models Section
    st.subheader("🎤 Whisper Models (Speech Recognition)")
    
    whisper_models = [
        ("tiny", "39 MB", "⚡ Fastest, basic quality", "Quick testing"),
        ("base", "142 MB", "⚖️ Balanced speed/quality", "General use"),
        ("small", "461 MB", "🎯 Better quality", "Better accuracy"),
        ("medium", "1.5 GB", "🏆 High quality", "Professional results"),
        ("large", "3.1 GB", "👑 Best quality", "Maximum accuracy")
    ]
    
    whisper_cols = st.columns(len(whisper_models))
    
    for i, (name, size, desc, use_case) in enumerate(whisper_models):
        with whisper_cols[i]:
            # Check if Whisper model is installed
            try:
                import whisper
                available_models = whisper.available_models()
                is_installed = name in available_models
            except ImportError:
                is_installed = False
            
            st.write(f"**{name.title()}**")
            st.write(f"📦 {size}")
            st.write(desc)
            st.write(f"🎯 {use_case}")
            
            if is_installed:
                st.success("✅ Installed")
            else:
                if st.button(f"⬇️ Install", key=f"whisper_{name}", use_container_width=True):
                    with st.spinner(f"Installing {name}..."):
                        try:
                            import whisper
                            model = whisper.load_model(name)
                            st.success(f"✅ {name} installed!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to install {name}: {str(e)}")
    
    st.divider()
    
    # Custom Model Download
    st.subheader("🛠️ Custom Model Download")
    
    with st.expander("🔧 Advanced: Download Custom Models"):
        st.write("For advanced users who want to download specific models from Hugging Face:")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            custom_model = st.text_input(
                "🤗 Hugging Face Model Repository",
                placeholder="e.g., microsoft/DialoGPT-medium, gpt2, distilbert-base-uncased",
                help="Enter any Hugging Face model repository name"
            )
            
            custom_dir = st.text_input(
                "📁 Local Directory",
                value="models/local_llm",
                help="Directory to save the model (relative to project root)"
            )
        
        with col2:
            st.write("**Examples:**")
            st.code("microsoft/DialoGPT-small", language="text")
            st.code("gpt2-large", language="text") 
            st.code("distilgpt2", language="text")
        
        if st.button("🚀 Download Custom Model", disabled=not custom_model, use_container_width=True):
            if custom_model:
                with st.spinner(f"📥 Downloading {custom_model}..."):
                    try:
                        cmd = ["huggingface-cli", "download", custom_model, "--local-dir", custom_dir]
                        Path(custom_dir).mkdir(parents=True, exist_ok=True)
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                        
                        if result.returncode == 0:
                            st.success(f"✅ Model {custom_model} downloaded successfully to {custom_dir}!")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to download {custom_model}: {result.stderr}")
                    except Exception as e:
                        st.error(f"❌ Error downloading {custom_model}: {str(e)}")
    
    # Current Models Status
    st.divider()
    st.subheader("📂 Installed Models")
    
    models_dir = Path("models")
    if models_dir.exists():
        model_subdirs = [d for d in models_dir.iterdir() if d.is_dir()]
        
        if model_subdirs:
            for model_dir in model_subdirs:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"📦 **{model_dir.name}**")
                    st.write(f"📁 `{model_dir}`")
                
                with col2:
                    try:
                        size_mb = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024*1024)
                        st.write(f"💾 {size_mb:.0f} MB")
                    except:
                        st.write("💾 Unknown")
                
                with col3:
                    files_count = len([f for f in model_dir.rglob('*') if f.is_file()])
                    st.write(f"📄 {files_count} files")
                
                with col4:
                    if st.button("🗑️", key=f"remove_dir_{model_dir.name}", help="Remove model"):
                        try:
                            shutil.rmtree(model_dir)
                            st.success(f"Model {model_dir.name} removed!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to remove: {e}")
        else:
            st.info("🔍 Models directory exists but no models found.")
    else:
        st.info("🔍 No models directory found. Models will be created when you download your first model.")
    
    # Quick Setup Guide
    st.divider()
    st.subheader("🚀 Quick Setup Guide")
    
    with st.expander("📖 First Time Setup - Click Here!"):
        st.markdown("""
        ### 🎯 Recommended Setup for Video Summarization:
        
        1. **🎬 For Movie Analysis (Recommended):**
           - Click **"🚀 One-Click Install"** for **Microsoft DialoGPT-medium**
           - This model excels at understanding dialogue and character interactions
           - Size: 1.2GB - good balance of quality and speed
        
        2. **🎤 For Speech Recognition:**
           - Install **Whisper Base** (142 MB) for general use
           - Or **Whisper Medium** (1.5 GB) for better quality
        
        3. **⚡ For Quick Testing:**
           - Install **DistilGPT2** (320 MB) for fast processing
           - Good for testing the system before using larger models
        
        ### 💡 Tips:
        - **GPU Users:** Can handle larger models (Medium/Large)
        - **CPU Users:** Stick with smaller models (Base/Small)
        - **Storage:** Ensure you have enough space before downloading
        - **Internet:** Large models may take time to download
        
        ### 📋 The Command You Requested:
        ```bash
        huggingface-cli download microsoft/DialoGPT-medium --local-dir models/local_llm
        ```
        This is now available as a one-click button above!
        """)
    
    # System Requirements Check
    with st.expander("💻 System Requirements & Compatibility"):
        st.write("### Current System Status:")
        
        # Check Python and dependencies
        try:
            import torch
            torch_version = torch.__version__
            torch_cuda = torch.cuda.is_available()
        except ImportError:
            torch_version = "Not installed"
            torch_cuda = False
        
        try:
            import transformers
            transformers_version = transformers.__version__
        except ImportError:
            transformers_version = "Not installed"
        
        req_col1, req_col2 = st.columns(2)
        
        with req_col1:
            st.write("**Software:**")
            st.write(f"🐍 Python: {sys.version.split()[0]}")
            st.write(f"🔥 PyTorch: {torch_version}")
            st.write(f"🤗 Transformers: {transformers_version}")
            st.write(f"🎮 CUDA Available: {'✅ Yes' if torch_cuda else '❌ No'}")
        
        with req_col2:
            st.write("**Hardware:**")
            try:
                available_storage = shutil.disk_usage(".")[2] / (1024**3)
                st.write(f"💾 Available Storage: {available_storage:.1f} GB")
            except:
                st.write("💾 Available Storage: Unknown")
            
            try:
                import psutil
                ram = psutil.virtual_memory().total / (1024**3)
                st.write(f"🧠 Total RAM: {ram:.1f} GB")
            except ImportError:
                st.write("🧠 Total RAM: Unknown")
                
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    st.write(f"🎮 GPU Memory: {gpu_memory:.1f} GB")
            except:
                pass
        
        # Recommendations based on system
        st.write("### 🎯 Personalized Recommendations:")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory > 8:
                    st.success("🎮 **High-end GPU detected!** You can run large models (DialoGPT-large, Whisper Large)")
                elif gpu_memory > 4:
                    st.info("🎯 **Mid-range GPU detected.** Recommended: DialoGPT-medium, Whisper Medium")
                else:
                    st.warning("💻 **Low GPU memory detected.** Recommended: DistilGPT2, Whisper Base/Small")
            else:
                st.warning("💻 **CPU processing detected.** Recommended: DistilGPT2, Whisper Base/Small")
        except:
            st.info("💻 **System analysis incomplete.** Start with lightweight models and upgrade as needed.")
    
    st.divider()
    
    # Action buttons at bottom
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Download Cache", use_container_width=True):
            try:
                import shutil
                cache_dir = Path.home() / ".cache" / "huggingface"
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                st.success("Hugging Face cache cleared!")
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")
    
    with col3:
        if st.button("📊 System Check", use_container_width=True):
            with st.spinner("Checking system dependencies..."):
                checks = []
                
                try:
                    import torch
                    checks.append("✅ PyTorch installed")
                except ImportError:
                    checks.append("❌ PyTorch not installed")
                
                try:
                    import transformers
                    checks.append("✅ Transformers installed")
                except ImportError:
                    checks.append("❌ Transformers not installed")
                
                try:
                    result = subprocess.run(["huggingface-cli", "--version"], capture_output=True)
                    if result.returncode == 0:
                        checks.append("✅ Hugging Face CLI available")
                    else:
                        checks.append("❌ Hugging Face CLI not available")
                except FileNotFoundError:
                    checks.append("❌ Hugging Face CLI not installed")
                
                for check in checks:
                    st.write(check)
                
                if all("✅" in check for check in checks):
                    st.success("System is ready for model downloads!")
                else:
                    st.warning("Some dependencies are missing. Install them for full functionality.")