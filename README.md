# Song Classify Project Setup

## For New Team Members (First Time Setup)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Supreme-Shrestha/songClassify.git
cd songClassify
```

### Step 2: Set Up Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install yt-dlp pandas matplotlib seaborn notebook
```

### Step 4: Configure Git (Optional but Recommended)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## Daily Workflow

### When You START Working
Pull the latest changes from the team:
```bash
python sync/sync_project.py start
```
Or:
```bash
source .venv/bin/activate  # If not already activated
python sync/sync_project.py start
```

### When You FINISH Working
Push your changes to share with the team:
```bash
python sync/sync_project.py end
```

---

## Project Scripts

### Add a New Genre Playlist
```bash
cd dataPrep
python create_jsonl.py
# It will ask for genre and link interactively

# OR with arguments:
python create_jsonl.py --genre "rock" --link "https://youtube.com/playlist?list=..."
```

### Download Audio Files
```bash
cd dataPrep
python download_audios.py
```
Note: This will download songs into the `data/` folder. The script tracks what's been downloaded, so you can run it multiple times safely.

### Analyze Playlists
Open the Jupyter notebook:
```bash
cd dataPrep
jupyter notebook playlist_analysis.ipynb
```

---

## Important Notes

- **Data is NOT synced**: The `data/` folder with audio files is ignored by git. Each team member downloads their own copy.
- **Always pull before starting**: Run `sync_project.py start` to get the latest code changes.
- **Always push when done**: Run `sync_project.py end` to share your changes.

---

## Troubleshooting

### "Not a git repository" error
You forgot to clone the repository. Go back to Step 1.

### "No remote repository configured"
You cloned the repo incorrectly. Delete the folder and clone again.

### Merge conflicts
If you get conflicts when pulling:
```bash
git status  # See which files have conflicts
# Edit the files to resolve conflicts
git add .
git rebase --continue
```

### Permission denied when pushing
Make sure you have access to the repository. Ask the repository owner to add you as a collaborator.
