import json
import os
import re
from itertools import groupby
from pydub import AudioSegment


def parse_textgrid(textgrid_path):
    with open(textgrid_path, "r", encoding="utf-8") as f:
        content = f.read()

    words = []
    intervals_section = re.search(r"intervals: size = \d+\s+(.*)", content, re.DOTALL)
    if not intervals_section:
        return words

    interval_pattern = (
        r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"'
    )
    matches = re.findall(interval_pattern, intervals_section.group(1))

    for xmin, xmax, text in matches:
        if text.strip():
            words.append(
                {"text": text.strip(), "start": float(xmin), "end": float(xmax)}
            )

    return words


def match_text_to_timing(speaker_text, textgrid_words):
    text_normalized = speaker_text.replace(" ", "")

    for start_idx in range(len(textgrid_words)):
        concatenated = ""
        for end_idx in range(start_idx, len(textgrid_words)):
            concatenated += textgrid_words[end_idx]["text"]

            if concatenated == text_normalized:
                return textgrid_words[start_idx : end_idx + 1]

            if len(concatenated) > len(text_normalized):
                break

    return []


directory_name = "shuka"
diarized_dir = f"diarized_data/{directory_name}"
processed_files = [
    f
    for f in os.listdir(diarized_dir)
    if f.startswith("processed_") and f.endswith(".txt")
]
max_file_num = max([int(f.split("_")[1].split(".")[0]) for f in processed_files])

for i in range(1, max_file_num + 1):
    file_path = f"diarized_data/{directory_name}/processed_{i}"
    text_file_path = file_path + ".txt"

    with open(text_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    audio_file_path = f"data/{directory_name}/processed_{i}.wav"
    audio = AudioSegment.from_wav(audio_file_path)

    textgrid_path = f"mfa/{directory_name}/batch_1_align/speaker_{i:03d}/audio.TextGrid"
    textgrid_words = parse_textgrid(textgrid_path)

    speaker_segments = []
    current_speaker = None
    current_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("audio_event:"):
            continue

        if line.startswith("speaker_"):
            if current_speaker is not None and current_text:
                combined_text = " ".join(current_text)
                matched_words = match_text_to_timing(combined_text, textgrid_words)

                if matched_words:
                    start_time = matched_words[0]["start"]
                    end_time = matched_words[-1]["end"]
                else:
                    start_time = 0
                    end_time = 0

                segment = {
                    "Text": combined_text,
                    "Starttime": start_time,
                    "Endtime": end_time,
                    "Speaker": current_speaker,
                }
                speaker_segments.append(segment)
                current_text = []

            parts = line.split(":", 1)
            current_speaker = parts[0].strip()
            if len(parts) > 1:
                current_text.append(parts[1].strip())
        else:
            if current_text:
                current_text.append(line)

    if current_speaker is not None and current_text:
        combined_text = " ".join(current_text)
        matched_words = match_text_to_timing(combined_text, textgrid_words)

        if matched_words:
            start_time = matched_words[0]["start"]
            end_time = matched_words[-1]["end"]
        else:
            start_time = 0
            end_time = 0

        segment = {
            "Text": combined_text,
            "Starttime": start_time,
            "Endtime": end_time,
            "Speaker": current_speaker,
        }
        speaker_segments.append(segment)

    turns_output_dir = f"turns/{directory_name}/processed_{i}/turns"
    os.makedirs(turns_output_dir, exist_ok=True)

    valid_turns = []
    j = 0
    while j < len(speaker_segments) - 2:
        speaker_a = speaker_segments[j]["Speaker"]
        speaker_b = speaker_segments[j + 1]["Speaker"]
        if speaker_a != speaker_b and speaker_segments[j + 2]["Speaker"] == speaker_a:
            turn_data = [
                speaker_segments[j],
                speaker_segments[j + 1],
                speaker_segments[j + 2],
            ]
            valid_turns.append(turn_data)
            j += 3
        else:
            j += 1

    if len(valid_turns) <= 2:
        print(
            f"Not enough turns found (only {len(valid_turns)} turns). Need at least 3 turns to exclude first and last."
        )
        continue

    turns_to_process = valid_turns[1:-1]

    for turn_count, turn_data in enumerate(turns_to_process, 1):
        last_turn_start = turn_data[2]["Starttime"]
        last_turn_end = turn_data[2]["Endtime"]
        start_time_ms = int(last_turn_start * 1000)
        end_time_ms = int(last_turn_end * 1000)

        json_filename = f"{turns_output_dir}/turn_{turn_count}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(turn_data, f, ensure_ascii=False, indent=4)

        audio_segment = audio[start_time_ms:end_time_ms]
        audio_filename = f"{turns_output_dir}/turn_{turn_count}.wav"
        audio_segment.export(audio_filename, format="wav")

        print(
            f"3-turn split {turn_count}: Last turn start time: {last_turn_start:.2f}s - Last turn end time: {last_turn_end:.2f}s"
        )
