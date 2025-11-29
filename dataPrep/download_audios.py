import json
import os
import yt_dlp

def download_audios(jsonl_path, output_base_dir, archive_file):
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
                continue
                
            genre = entry.get('genre')
            link = entry.get('link')

            if not genre or not link:
                print(f"Skipping invalid entry: {line}")
                continue

            # Clean the link (remove backslashes used for escaping in the file)
            link = link.replace('\\', '')
            
            print(f"Processing genre: {genre}")

            genre_dir = os.path.join(output_base_dir, genre)
            if not os.path.exists(genre_dir):
                os.makedirs(genre_dir)

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(genre_dir, '%(title)s.%(ext)s'),
                'download_archive': archive_file, # This tracks downloaded videos to prevent re-downloading
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'ignoreerrors': True,
                'quiet': False,
                'no_warnings': True,
                'cookiefile': os.path.join(os.path.dirname(jsonl_path), 'cookies.txt'),
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([link])
            except Exception as e:
                print(f"Error downloading {link}: {e}")

if __name__ == "__main__":
    # Paths are relative to this script's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_file = os.path.join(current_dir, "my_data.jsonl")
    # Output data to ../data
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    # Archive file to track downloads
    archive_file = os.path.join(current_dir, "download_archive.txt")
    
    print(f"Reading from: {jsonl_file}")
    print(f"Saving to: {data_dir}")
    print(f"Archive file: {archive_file}")
    
    download_audios(jsonl_file, data_dir, archive_file)
