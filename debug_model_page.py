#!/usr/bin/env python3
"""
Debug script to test Model Management page directly
"""
import streamlit as st
from pathlib import Path
import sys

# Add root path to access modules
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))

st.set_page_config(
    page_title="Model Management Debug",
    page_icon="🧠",
    layout="wide"
)

# Import and run the model manager directly
try:
    from pages.model_manager_enhanced import show_model_manager
    st.title("🧠 Model Management - Direct Test")
    show_model_manager()
except Exception as e:
    st.error(f"Error loading Model Manager: {e}")
    import traceback
    st.code(traceback.format_exc())