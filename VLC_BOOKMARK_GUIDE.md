# VLC Bookmark Usage Guide

## How to Access Bookmarks in VLC

Your video summarization app creates **XSPF playlist files** with bookmarks for key movie scenes. Here's how to use them:

### Method 1: Open Playlist File Directly
1. **Find your bookmark file**: Located in `output/bookmarks/` 
   - File name: `[MovieName]_bookmarks_[timestamp].xspf`
2. **Double-click the .xspf file** - it should open in VLC automatically
3. **Or drag & drop** the .xspf file into VLC

### Method 2: Open in VLC Manually
1. **Open VLC**
2. **Media menu** → **Open File** (Ctrl+O)
3. **Navigate to** `output/bookmarks/` folder
4. **Select your .xspf file**
5. **Click Open**

## What You'll See in VLC

### Playlist View
- **View menu** → **Playlist** (Ctrl+L) to see the playlist panel
- Each bookmark appears as a **separate track** with descriptions like:
  - "Scene 1: Character Introduction"
  - "Scene 2: Major Plot Point" 
  - "Scene 3: Climax Moment"

### Navigation
- **Click any scene** in the playlist to jump directly to that timestamp
- **Double-click** to play from that specific scene
- Each bookmark includes **start time** and **duration** information
- **Next/Previous buttons** jump between key scenes

## Current Bookmark Features

✅ **Automatic Scene Detection**: Key narrative moments identified  
✅ **Timestamp Precision**: Exact start/end times for each scene  
✅ **Scene Descriptions**: AI-generated descriptions of what happens  
✅ **Original Video Integration**: Bookmarks reference your original movie file  

## Troubleshooting

### "File not found" in VLC
- Make sure your **original video file** is still in the same location
- The bookmark file references the original video path

### Bookmarks don't appear
- Use **View menu** → **Playlist** to see the bookmark list
- Make sure you're opening the **.xspf file**, not the video directly

### No bookmark files created
- This means video processing hasn't completed successfully yet
- Check the app processing status and look for any error messages

## Alternative: Manual VLC Bookmarks

If you prefer VLC's built-in bookmarks:
1. **Play your video** in VLC
2. **At important moments**: **Playback menu** → **Custom Bookmarks** → **Manage**
3. **Click "Create"** to add bookmark at current time
4. **Name your bookmark** (e.g., "Opening Scene", "Climax")
5. **Access later**: **Playback menu** → **Custom Bookmarks** → [Your bookmark name]

## Future Enhancements

The app could be enhanced to also create:
- **VLC bookmark (.xbel)** files for native VLC bookmark integration
- **Chapter markers** embedded directly in video files
- **Web-based bookmark player** for browser viewing