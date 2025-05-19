import os
import subprocess
import re
import concurrent.futures

audio_format = "wav"

lang_files = {
    "english": "lists/english.txt",
    "korean": "lists/korean.txt",
    # "japanese": "lists/japanese.txt",
    # "chinese": "lists/chinese.txt",
}


def sanitize_folder_name(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)


def download_playlist(lang, url, playlist_index):
    folder_name = f"playlist_{playlist_index}"
    output_dir = f"downloads/{lang}/{folder_name}"
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "yt-dlp",
        url,
        "-x",
        "--audio-format",
        audio_format,
        "--audio-quality",
        "0",
        "--output",
        os.path.join(output_dir, "%(title)s.%(ext)s"),
        "--ignore-errors",
        "--no-overwrites",
        "--cookies",
        "cookies.txt",
    ]

    print(f"Downloading {folder_name} for language: {lang}")
    subprocess.run(command)


def main():
    download_tasks = []
    
    for lang, txt_path in lang_files.items():
        with open(txt_path, "r") as f:
            playlist_urls = [line.strip() for line in f if line.strip()]

        for i, url in enumerate(playlist_urls):
            playlist_index = i + 1
            download_tasks.append((lang, url, playlist_index))
    
    max_workers = 50
    print(f"Using {max_workers} CPU cores for parallel downloads")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_playlist, lang, url, index) for lang, url, index in download_tasks]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    main()
