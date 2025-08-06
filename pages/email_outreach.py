"""
Email Outreach page for MovieDigest AI marketing system
"""
import streamlit as st
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from marketing.email_outreach_system import show_email_outreach_page

def main():
    """Main email outreach page."""
    show_email_outreach_page()

if __name__ == "__main__":
    main()