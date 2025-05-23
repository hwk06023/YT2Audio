import os
import subprocess
import sys
from textgrids import TextGrid
import torchaudio
import re

directory_name = "짚톡 ) MBTI 토론 1회 (w.핑맨,김똘복,꽃핀)"  # enter directory name

for i in range(1, len(os.listdir("data/" + directory_name)) + 1):
    file_path = "data/" + directory_name + "/processed_" + str(i)
    text_file_path = "transcription_" + file_path + ".txt"
    audio_file_path = file_path + ".wav"

    audio_name = audio_file_path.split("/")[-1].split(".wav")[0]
    data_directory = f"mfa/{directory_name}/{audio_name}"
    align_directory = f"mfa/{directory_name}/{audio_name}_align"
    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(align_directory, exist_ok=True)

    lab_file_path = f"{data_directory}/audio.lab"
    lab_file_dir = os.path.dirname(lab_file_path)
    if not os.path.exists(lab_file_dir):
        os.makedirs(lab_file_dir, exist_ok=True)
    if not os.path.exists(lab_file_path):
        with open(lab_file_path, "w") as f:
            if os.path.exists(text_file_path):
                text = open(text_file_path, "r").read()
                print("text:", text)
                words = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)
                removed_words = [
                    word for word in text.split() if word not in words.split()
                ]
                if removed_words:
                    print(f"Removed words from {text_file_path}: {removed_words}")
                f.write(" ".join(words))
            else:
                print(f"Text file not found: {text_file_path}")
    else:
        with open(lab_file_path, "r") as f:
            text = f.read()

    textgrid_file_path = f"{align_directory}/audio.TextGrid"
    # Ensure align directory exists
    if not os.path.exists(align_directory):
        os.makedirs(align_directory, exist_ok=True)

    if not os.path.exists(textgrid_file_path):
        # Create wav file directory if needed
        wav_output_path = f"{data_directory}/audio.wav"
        wav_output_dir = os.path.dirname(wav_output_path)
        if not os.path.exists(wav_output_dir):
            os.makedirs(wav_output_dir, exist_ok=True)

        # Validate the audio and text corpus first
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio_file_path,
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                wav_output_path,
            ]
        )
        subprocess.run(
            [
                "mfa",
                "validate",
                data_directory,
                "korean_mfa",
                "--clean",
                "--beam",
                "40000",
            ]
        )
        print("mfa validate done")

        # Use MFA to get .TextGrid format
        subprocess.run(
            [
                "mfa",
                "align",
                data_directory,
                "korean_mfa",
                "korean_mfa",
                align_directory,
            ]
        )
        print("mfa align done")
    else:
        print(f"TextGrid file already exists: {textgrid_file_path}")
