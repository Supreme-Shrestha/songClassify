# Setup Instructions for Lightning.ai

This document explains how to set up and run the `download_audios.py` script on Lightning.ai to avoid YouTube bot detection errors.

## Problem
The original error occurred because YouTube detected automated access:
```
ERROR: [youtube] Sign in to confirm you're not a bot
```

## Solution
The script has been updated to use cookies from your browser, which authenticates your YouTube session.

## Setup Steps

### 1. Export YouTube Cookies (Do this on your local machine with a browser)

You need to export cookies from a browser where you're logged into YouTube:

**Option A: Using yt-dlp directly (Recommended)**
```bash
# This will extract cookies from Firefox and save them to cookies.txt
yt-dlp --cookies-from-browser firefox --cookies dataPrep/cookies.txt --skip-download "https://www.youtube.com"
```

**Option B: Using a browser extension**
- Install "Get cookies.txt LOCALLY" extension for Chrome/Firefox
- Go to youtube.com (make sure you're logged in)
- Click the extension and export cookies to `cookies.txt`
- Move the file to `dataPrep/cookies.txt`

### 2. Transfer Files to Lightning.ai

Once you have the SSH connection set up, transfer the necessary files:

```bash
# Transfer the cookies file
scp dataPrep/cookies.txt s_01kazj1pkp0sy7x8v9ssv78bq1@ssh.lightning.ai:content/songClassify/dataPrep/

# Transfer the updated script (if needed)
scp dataPrep/download_audios.py s_01kazj1pkp0sy7x8v9ssv78bq1@ssh.lightning.ai:content/songClassify/dataPrep/
```

### 3. Set Up Python Environment on Lightning.ai

```bash
# SSH into Lightning.ai
ssh s_01kazj1pkp0sy7x8v9ssv78bq1@ssh.lightning.ai

# Create a conda environment in the project directory
/system/conda/miniconda3/bin/conda create -p ~/content/songClassify/.conda_env python=3.12 -y

# Install yt-dlp
~/content/songClassify/.conda_env/bin/pip install yt-dlp
```

### 4. Run the Script

```bash
# Navigate to the dataPrep directory
cd content/songClassify/dataPrep

# Run the script using the conda environment
/teamspace/studios/this_studio/content/songClassify/.conda_env/bin/python download_audios.py
```

## What Changed in the Code

The `download_audios.py` script now includes:
```python
'cookiefile': os.path.join(os.path.dirname(jsonl_path), 'cookies.txt'),
```

This tells yt-dlp to use the cookies file for authentication, which prevents the bot detection error.

## Important Notes

1. **Cookie Expiration**: Cookies may expire after some time. If you get bot detection errors again, re-export fresh cookies.

2. **Privacy**: The `cookies.txt` file contains your authentication data. Add it to `.gitignore` to avoid committing it to version control.

3. **Browser Choice**: The script now uses a cookies file instead of directly accessing browser cookies, so it works on servers without browsers installed.

4. **Environment Path**: The conda environment is created in the project directory (`.conda_env`) to ensure it persists across Lightning.ai sessions.

## Troubleshooting

**Error: "failed to load cookies"**
- Make sure `cookies.txt` exists in the `dataPrep` directory
- Re-export cookies from your browser

**Error: "Sign in to confirm you're not a bot"**
- Your cookies may have expired
- Export fresh cookies from a browser where you're logged into YouTube

**Error: "ModuleNotFoundError: No module named 'yt_dlp'"**
- Make sure you're using the correct Python interpreter from the conda environment
- Re-run the pip install command
