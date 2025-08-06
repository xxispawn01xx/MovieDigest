#!/usr/bin/env python3
"""
Script to download LLM models for the video summarization project.
This will download models to the correct directory structure.
"""
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def download_recommended_model():
    """Download a recommended model for narrative analysis."""
    
    print("🤖 Video Summarization LLM Model Downloader")
    print("=" * 50)
    
    # Show model options
    models = {
        "1": {
            "name": "distilgpt2",
            "size": "~320MB",
            "description": "Lightweight, good for basic narrative analysis"
        },
        "2": {
            "name": "gpt2-medium", 
            "size": "~1.5GB",
            "description": "Better quality, balanced size/performance"
        },
        "3": {
            "name": "microsoft/DialoGPT-medium",
            "size": "~1.2GB", 
            "description": "Specialized for dialogue understanding"
        },
        "4": {
            "name": "microsoft/DialoGPT-large",
            "size": "~3GB",
            "description": "High quality dialogue/narrative analysis"
        }
    }
    
    print("\nAvailable Models:")
    for key, model in models.items():
        print(f"  {key}. {model['name']}")
        print(f"     Size: {model['size']}")
        print(f"     Description: {model['description']}\n")
    
    choice = input("Choose a model (1-4) or 'skip' to see manual instructions: ").strip()
    
    if choice.lower() == 'skip':
        show_manual_instructions()
        return
        
    if choice not in models:
        print("❌ Invalid choice")
        return
        
    selected_model = models[choice]["name"]
    print(f"\n📥 Selected: {selected_model}")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Create models directory
        models_dir = Path("models/local_llm")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 Downloading to: {models_dir.absolute()}")
        print("⏳ This may take a few minutes depending on model size...")
        
        # Download model
        snapshot_download(
            repo_id=selected_model,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Successfully downloaded {selected_model}")
        print(f"📍 Model saved to: {models_dir.absolute()}")
        print("\n🎬 Your video summarization app can now use LLM-powered narrative analysis!")
        
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("📦 Install it with: pip install huggingface_hub")
        print("🔄 Or use the manual download instructions below:")
        show_manual_instructions()
    except Exception as e:
        print(f"❌ Download failed: {e}")
        show_manual_instructions()

def show_manual_instructions():
    """Show manual download instructions."""
    
    print("\n📋 Manual Download Instructions")
    print("=" * 35)
    
    print("1️⃣ **Option A: Using Git LFS (Recommended)**")
    print("   cd models")
    print("   git clone https://huggingface.co/distilgpt2 local_llm")
    
    print("\n2️⃣ **Option B: Using Hugging Face CLI**")
    print("   pip install huggingface_hub")
    print("   huggingface-cli download distilgpt2 --local-dir models/local_llm")
    
    print("\n3️⃣ **Option C: Using Python Script**")
    print("   from huggingface_hub import snapshot_download")
    print("   snapshot_download('distilgpt2', local_dir='models/local_llm')")
    
    print("\n📁 **Directory Structure After Download:**")
    print("   models/")
    print("   └── local_llm/")
    print("       ├── config.json")
    print("       ├── pytorch_model.bin")
    print("       ├── tokenizer.json")
    print("       └── tokenizer_config.json")
    
    print("\n🔗 **Recommended Model URLs:**")
    print("   • distilgpt2: https://huggingface.co/distilgpt2")
    print("   • gpt2-medium: https://huggingface.co/gpt2-medium") 
    print("   • DialoGPT-medium: https://huggingface.co/microsoft/DialoGPT-medium")
    
    print("\n⚡ **For RTX 3060 Users:**")
    print("   All these models will work great with GPU acceleration!")

if __name__ == "__main__":
    download_recommended_model()