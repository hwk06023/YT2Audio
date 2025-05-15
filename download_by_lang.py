import os
import subprocess
import re

audio_format = "wav"

lang_files = {
    "english": "lists/english.txt",
    "korean": "lists/korean.txt",
    # "japanese": "lists/japanese.txt",
    # "chinese": "lists/chinese.txt",
}


def sanitize_folder_name(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)


for lang, txt_path in lang_files.items():
    with open(txt_path, "r") as f:
        playlist_urls = [line.strip() for line in f if line.strip()]

    for i, url in enumerate(playlist_urls):
        playlist_index = i + 1

        # 플레이리스트 이름 얻기
        print(f"Getting playlist info for {url}...")
        info_command = [
            "yt-dlp",
            "--skip-download",
            "--print",
            "%(playlist_title)s",
            url,
        ]

        result = subprocess.run(info_command, capture_output=True, text=True)
        playlist_title = result.stdout.strip()

        if playlist_title:
            folder_name = sanitize_folder_name(playlist_title)
        else:
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
        ]

        print(f"Downloading {folder_name} for language: {lang}")
        subprocess.run(command)
