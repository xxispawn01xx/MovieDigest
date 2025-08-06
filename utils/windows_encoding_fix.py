"""
Windows Command Prompt encoding fix for Unicode issues
Handles cases where subprocess calls fail due to Windows cp1252 encoding limitations
"""
import subprocess
import os
import sys
from pathlib import Path

def run_command_with_encoding_fix(cmd, **kwargs):
    """
    Run subprocess command with Windows encoding fixes applied.
    
    Args:
        cmd: Command to run (list or string)
        **kwargs: Additional subprocess.run arguments
    
    Returns:
        subprocess.CompletedProcess result
    """
    # Set up environment to handle Unicode properly on Windows
    env = os.environ.copy()
    
    # Force UTF-8 encoding for Python I/O
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # Enable legacy Windows stdio handling
    env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
    
    # Disable color output that might cause encoding issues
    env['NO_COLOR'] = '1'
    env['TERM'] = 'dumb'
    
    # Set default encoding parameters if not provided
    encoding_kwargs = {
        'encoding': 'utf-8',
        'errors': 'ignore',  # Ignore characters that can't be encoded
        'env': env
    }
    
    # Update with user-provided kwargs
    encoding_kwargs.update(kwargs)
    
    try:
        return subprocess.run(cmd, **encoding_kwargs)
    except UnicodeEncodeError as e:
        # If still having encoding issues, try with ASCII fallback
        print(f"Unicode encoding error: {e}")
        print("Retrying with ASCII encoding...")
        
        encoding_kwargs['encoding'] = 'ascii'
        encoding_kwargs['errors'] = 'replace'
        
        return subprocess.run(cmd, **encoding_kwargs)

def fix_huggingface_cli_download(model_repo, local_dir, timeout=1800):
    """
    Download Hugging Face model with Windows encoding fixes.
    
    Args:
        model_repo: Hugging Face model repository name
        local_dir: Local directory to save model
        timeout: Download timeout in seconds
    
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    try:
        # Create directory if it doesn't exist
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # Build command - use hf download (new command) instead of deprecated huggingface-cli download
        cmd = ["python", "-m", "huggingface_hub.commands.huggingface_cli", "download", model_repo, "--local-dir", local_dir]
        
        # Try new command first (hf download)
        new_cmd = ["hf", "download", model_repo, "--local-dir", local_dir]
        
        print(f"Attempting to download {model_repo} to {local_dir}")
        print(f"Command: {' '.join(new_cmd)}")
        
        # Try new hf command first
        result = run_command_with_encoding_fix(
            new_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            print("New 'hf' command failed, trying legacy 'huggingface-cli'...")
            print(f"Error was: {result.stderr}")
            
            # Fallback to legacy command
            result = run_command_with_encoding_fix(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        
        if result.returncode == 0:
            print(f"Successfully downloaded {model_repo}")
            return True, None
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            print(f"Download failed: {error_msg}")
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        return False, f"Download timed out after {timeout} seconds"
    except FileNotFoundError as e:
        if "hf" in str(e):
            return False, "Hugging Face CLI not found. Please install with: pip install huggingface_hub[cli]"
        else:
            return False, f"Command not found: {e}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def test_encoding_fix():
    """Test the encoding fix with a simple command."""
    try:
        # Test with a command that might have Unicode output
        result = run_command_with_encoding_fix(
            ["python", "--version"],
            capture_output=True,
            text=True
        )
        
        print(f"Test result: {result.returncode}")
        print(f"Output: {result.stdout}")
        
        return result.returncode == 0
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Windows encoding fix...")
    if test_encoding_fix():
        print("✅ Encoding fix working properly")
    else:
        print("❌ Encoding fix test failed")