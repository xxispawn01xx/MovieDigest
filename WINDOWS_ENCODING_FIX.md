# Windows Unicode Encoding Fix

## Problem
Windows Command Prompt (cp1252 encoding) can't display Unicode characters like emojis (⚠️) used by Hugging Face CLI, causing:
```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 5-6: character maps to <undefined>
```

## Solution Applied
Created `utils/windows_encoding_fix.py` with comprehensive Unicode handling:

### Key Fixes
1. **Environment Variables**: Force UTF-8 encoding
   ```python
   env['PYTHONIOENCODING'] = 'utf-8'
   env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
   ```

2. **Command Fallback**: Try modern `hf download` command first, fallback to legacy `huggingface-cli download`

3. **Encoding Parameters**: Handle Unicode gracefully
   ```python
   encoding='utf-8',
   errors='ignore'  # Skip problematic characters
   ```

4. **ASCII Fallback**: If UTF-8 fails, retry with ASCII encoding

## Updated Files
- `pages/model_manager_enhanced.py`: Uses new encoding-safe download function
- `utils/windows_encoding_fix.py`: Centralized encoding fix utility

## Usage
The model downloader now automatically handles Windows encoding issues when downloading Mistral models or any Hugging Face model.

## Test Command
To verify the fix works:
```bash
python utils/windows_encoding_fix.py
```

This should show: `✅ Encoding fix working properly`

## Alternative Manual Fix
If issues persist, you can set these environment variables before running:
```cmd
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=1
streamlit run app.py --server.port 5000
```