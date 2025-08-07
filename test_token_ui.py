#!/usr/bin/env python3
"""
Quick test to verify the token UI is working
"""
import streamlit as st
import os
from pathlib import Path

st.title("🔑 HF Token Test")

# Test the exact same token UI code
st.subheader("🔑 Hugging Face Configuration")

# Check if token already exists
current_token = os.environ.get('HF_TOKEN', '')

col1, col2 = st.columns([3, 1])

with col1:
    hf_token = st.text_input(
        "Hugging Face Token (for one-click model downloads):",
        value="***Hidden***" if current_token else "",
        type="password" if current_token else "default",
        help="Get your token from https://huggingface.co/settings/tokens"
    )

with col2:
    if st.button("💾 Save Token", use_container_width=True):
        if hf_token and hf_token != "***Hidden***":
            # Save to environment for current session
            os.environ['HF_TOKEN'] = hf_token
            st.success("Token saved for this session!")
            st.rerun()
        elif not hf_token:
            st.error("Please enter a valid token")

if current_token:
    st.success("✅ Hugging Face token is configured")
    if st.button("🗑️ Clear Token"):
        os.environ.pop('HF_TOKEN', None)
        st.info("Token cleared")
        st.rerun()
else:
    st.info("💡 Add your Hugging Face token for seamless model downloads")
    st.markdown("""
    **How to get your Hugging Face token:**
    1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    2. Create a new token with 'Read' permissions
    3. Copy and paste it above
    """)

st.write("Current environment HF_TOKEN:", "Set" if current_token else "Not set")