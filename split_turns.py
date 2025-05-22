import json
import os
from itertools import groupby
from pydub import AudioSegment

directory_name = ""

for i in range(1, len(os.listdir("data/" + directory_name)) + 1):
    file_path = "data/" + directory_name + "/processed_" + str(i)

    # TODO: use enhanced txt files by using MFA(Montreal Forced Aligner)
    text_file_path = file_path + ".txt"
    result = open(text_file_path, "r", encoding="utf-8").read()

    audio_file_path = file_path + ".wav"
    audio = AudioSegment.from_wav(audio_file_path)

    speaker_segments = []
    word_segments = []

    word_only = [word for word in result.words if word.type == "word"]

    for speaker_id, group in groupby(word_only, key=lambda x: x.speaker_id):
        group_list = list(group)
        combined_text = " ".join([word.text for word in group_list])
        start_time = group_list[0].start
        end_time = group_list[-1].end

        segment = {
            "Text": combined_text,
            "Starttime": start_time,
            "Endtime": end_time,
            "Speaker": speaker_id,
        }

        speaker_segments.append(segment)

    # Find patterns of A-B-A
    turns_output_dir = f"data/{directory_name}" + "/processed_" + str(i + 1) + "/turns"
    if not os.path.exists(turns_output_dir):
        os.makedirs(turns_output_dir)

    turn_count = 0

    # Find patterns of A-B-A
    i = 0
    while i < len(speaker_segments) - 2:
        speaker_a = speaker_segments[i]["Speaker"]
        speaker_b = speaker_segments[i + 1]["Speaker"]

        # Check for pattern A-B-A
        if speaker_a != speaker_b and speaker_segments[i + 2]["Speaker"] == speaker_a:
            turn_data = [
                speaker_segments[i],
                speaker_segments[i + 1],
                speaker_segments[i + 2],
            ]

            # Time range of 3 turns
            start_time_sec = turn_data[0]["Starttime"]
            end_time_sec = turn_data[2]["Endtime"]

            # Convert to milliseconds
            start_time_ms = int(start_time_sec * 1000)
            end_time_ms = int(end_time_sec * 1000)

            # Consider offset(60 seconds) and calculate the position in the original audio
            original_start_ms = start_time_ms + start_time
            original_end_ms = end_time_ms + start_time

            # Save JSON file
            turn_count += 1
            json_filename = f"{turns_output_dir}/turn_{turn_count}.json"

            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(turn_data, f, ensure_ascii=False, indent=4)

            # Cut audio segment and save
            audio_segment = audio[original_start_ms:original_end_ms]
            audio_filename = f"{turns_output_dir}/turn_{turn_count}.wav"
            audio_segment.export(audio_filename, format="wav")

            print(
                f"3-turn split {turn_count}: Time: {start_time_sec:.2f}s - {end_time_sec:.2f}s"
            )

            # Move to next search position (skip already used turn)
            i += 3
        else:
            # If pattern doesn't match, move to next position
            i += 1
