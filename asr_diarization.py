from elevenlabs import ElevenLabs
import os
import traceback
from dotenv import load_dotenv
from pydub import AudioSegment
import io
import argparse
import json
from itertools import groupby
from time import sleep
import httpx
import re

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    timeout=1200,
)


def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    return file_path


def process_audio_file(
    audio_file_path, output_prefix="", apply_audio_processing=True, is_main_file=True
):
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]

    if is_main_file:
        name = base_name
        transcription_file = f"transcription_data/{name}.txt"
        diarized_file = f"diarized_data/{name}.txt"
    else:
        name = os.path.basename(os.path.dirname(audio_file_path))
        transcription_file = f"transcription_data/{name}/{base_name}.txt"
        diarized_file = f"diarized_data/{name}/{base_name}.txt"
    if os.path.exists(transcription_file) and os.path.exists(diarized_file):
        print(f"Skipping {audio_file_path} - already processed")
        return True

    print(f"Processing: {audio_file_path}")

    audio_file_path = f"data/{audio_file_path}"
    if not os.path.exists(audio_file_path):
        print(f"File does not exist: {audio_file_path}")
        return False

    audio = AudioSegment.from_wav(audio_file_path)

    if is_main_file:
        start_time = 60 * 1000
        end_time = len(audio) - 60 * 1000
        audio = audio[start_time:end_time]

    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)

    print(f"Converting speech to text...")
    print("Diarization: enabled")

    result = client.speech_to_text.convert(
        file=buffer, model_id="scribe_v1", diarize=True, tag_audio_events=False
    )

    print("\n--- Transcription Result ---")
    print(result.text)

    ensure_dir_exists(transcription_file)

    transcription_text = ""
    if hasattr(result, "words") and result.words:
        word_texts = []
        for word in result.words:
            if word.type == "word":
                if word.text:
                    word_texts.append(word.text)
        transcription_text = " ".join(word_texts)
    else:
        transcription_text = result.text

    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write(transcription_text)

    output_file = diarized_file
    text_to_save = ""
    if hasattr(result, "words") and result.words:
        formatted_text_parts = []
        current_speaker = None
        line_buffer = ""

        for word in result.words:
            if word.type == "word":
                speaker = word.speaker_id if hasattr(word, "speaker_id") else "Unknown"

                if word.text:
                    if speaker != current_speaker:
                        if line_buffer:
                            formatted_text_parts.append(line_buffer.strip())
                        line_buffer = f"{speaker}: {word.text}"
                        current_speaker = speaker
                    else:
                        if line_buffer:
                            line_buffer += f" {word.text}"
                        else:
                            line_buffer = f"{speaker}: {word.text}"
            elif word.type == "audio_event":
                if line_buffer:
                    formatted_text_parts.append(line_buffer.strip())
                    line_buffer = ""
                    current_speaker = None
                formatted_text_parts.append(f"audio_event: {word.text}")

        if line_buffer:
            formatted_text_parts.append(line_buffer.strip())

        text_to_save = "\n".join(formatted_text_parts)
    else:
        text_to_save = result.text

    ensure_dir_exists(output_file)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text_to_save)

    if os.path.exists(output_file):
        print(f"Transcription result saved to {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print(f"File not created: {output_file}")

    if hasattr(result, "words") and result.words:
        print("\nWords with timing and speaker info:")
        speakers = set()
        audio_events = []

        for word in result.words:
            if word.type == "audio_event":
                print(f"- Audio event: {word.text} at {word.start}s to {word.end}s")
                audio_events.append(
                    {
                        "text": word.text,
                        "start": word.start,
                        "end": word.end,
                        "type": "audio_event",
                    }
                )
            elif word.type == "word":
                if word.text:
                    print(
                        f"- Word: '{word.text}' (Speaker {word.speaker_id}) at {word.start}s to {word.end}s"
                    )
                    speakers.add(word.speaker_id)
        print(f"\nTotal speakers detected: {len(speakers)}")
        for speaker in speakers:
            print(f"- {speaker}")

    return True


directory_name = "mammoth"

main_audio_file = f"{directory_name}.wav"
process_audio_file(main_audio_file, apply_audio_processing=True, is_main_file=True)

data_directory = f"data/{directory_name}"
if os.path.exists(data_directory):
    for i in range(1, len(os.listdir(data_directory)) + 1):
        processed_audio_file = f"{directory_name}/processed_{i}.wav"
        process_audio_file(
            processed_audio_file, apply_audio_processing=False, is_main_file=False
        )
else:
    print(f"Directory does not exist: {data_directory}")
