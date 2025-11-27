import json
import os
import argparse
import pandas as pd

def read_jsonl_file(filepath):
    """
    reads all entries from a JSONL file and returns them as a list of dictionaries.
    """
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"warning: skipping malformed line in {filepath}: {line.strip()}")
    return data

def write_jsonl_file(filepath, data_list):
    """
    writes a list of dictionaries to a JSONL file, overwriting existing content.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True) # ensure directory exists
    with open(filepath, 'w') as f:
        for entry in data_list:
            json.dump(entry, f)
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="add data to a JSONL file and deduplicate by genre.")
    parser.add_argument("--genre", help="the genre of the data.")
    parser.add_argument("--link", help="the link associated with the genre.")
    args = parser.parse_args()

    genre = args.genre
    link = args.link

    # If arguments are not provided, ask for input interactively
    if not genre:
        genre = input("Enter genre: ").strip()
    if not link:
        link = input("Enter link: ").strip()

    if not genre or not link:
        print("Error: Both genre and link are required.")
        exit(1)

    jsonl_file_path = "dataPrep/my_data.jsonl" # now the jsonl file will be inside the dataPrep folder

    # ensure the directory for the jsonl file exists
    os.makedirs(os.path.dirname(jsonl_file_path), exist_ok=True)

    # 1. read existing data
    existing_data = read_jsonl_file(jsonl_file_path)

    # 2. add the new data
    new_entry = {"genre": genre, "link": link}
    all_data = existing_data + [new_entry]

    # 3. deduplicate by genre, keeping the last (most recent) entry
    if all_data:
        df = pd.DataFrame(all_data)
        # keep='last' ensures that if a genre is duplicated, the newly added or last encountered entry is kept.
        deduplicated_df = df.drop_duplicates(subset=['genre'], keep='last')
        final_data = deduplicated_df.to_dict(orient='records')
    else:
        final_data = []

    # 4. write the deduplicated data back to the file
    write_jsonl_file(jsonl_file_path, final_data)

    print(f"added/updated data for genre '{genre}' to {jsonl_file_path}.")
    print("file content after update:")
    with open(jsonl_file_path, 'r') as f:
        print(f.read())
    print(f"\nyou can view the content of '{jsonl_file_path}' by running: cat {jsonl_file_path}")