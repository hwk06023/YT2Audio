import os
import subprocess

audio_format = "wav"

lang_files = {
    "english": "lists/english.txt",
    "korean": "lists/korean.txt",
    # "japanese": "lists/japanese.txt",
    # "chinese": "lists/chinese.txt",
}

for lang, txt_path in lang_files.items():
    output_dir = f"downloads/{lang}"
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "yt-dlp",
        "-a",
        txt_path,
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

    print(f"Downloading for language: {lang}")
    subprocess.run(command)
