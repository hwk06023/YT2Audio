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

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    timeout=300,  # Increasing timeout to 5 minutes
)


def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    return file_path


directory_name = "짚톡 ) MBTI 토론 1회 (w.핑맨,김똘복,꽃핀)"  # enter directory name

for i in range(1, len(os.listdir("data/" + directory_name)) + 1):
    file_path = "data/" + directory_name + "/processed_" + str(i)
    text_file_path = file_path + ".txt"
    metadata_file_path = "meta" + text_file_path
    audio_file_path = file_path + ".wav"

    # Create metadata directory if it doesn't exist
    ensure_dir_exists(metadata_file_path)

    if os.path.exists(audio_file_path):
        audio = AudioSegment.from_wav(audio_file_path)
    else:
        print(f"File does not exist: {audio_file_path}")
        continue

    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)

    print(f"{i}th file: converting speech to text...")
    print("Diarization: enabled")

    result = client.speech_to_text.convert(
        file=buffer,
        model_id="scribe_v1",
        tag_audio_events=True,
        diarize=True,
        num_speakers=4,  # TODO: change this value based on the number of speakers
        timestamps_granularity="none",
    )

    print("\n--- Transcription Result ---")
    print(result.text)

    # Save transcription result (excluding audio events)
    transcription_file = "transcription_" + text_file_path
    ensure_dir_exists(transcription_file)

    # Extract only words, excluding audio events
    transcription_text = ""
    if hasattr(result, "words") and result.words:
        word_texts = []
        for word in result.words:
            if word.type == "word":
                word_texts.append(word.text)
        transcription_text = " ".join(word_texts)
    else:
        transcription_text = result.text

    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write(transcription_text)

    # Save diarized text
    output_file = "diarized_" + text_file_path
    text_to_save = ""
    if hasattr(result, "words") and result.words:
        formatted_text_parts = []
        current_speaker = None
        line_buffer = ""

        for word in result.words:
            if word.type == "word":
                speaker = word.speaker_id if hasattr(word, "speaker_id") else "Unknown"
                if speaker != current_speaker:
                    if line_buffer:  # Append previous line buffer
                        formatted_text_parts.append(line_buffer.strip())
                    line_buffer = f"{speaker}: {word.text}"  # Start new line buffer
                    current_speaker = speaker
                else:
                    if line_buffer:  # Add space only if buffer has content
                        line_buffer += f" {word.text}"
                    else:  # If buffer somehow empty, start it
                        line_buffer = f"{speaker}: {word.text}"

            elif word.type == "audio_event":
                if line_buffer:  # Append previous line buffer before event
                    formatted_text_parts.append(line_buffer.strip())
                    line_buffer = ""  # Reset buffer
                    current_speaker = None  # Reset speaker context for events
                formatted_text_parts.append(f"audio_event: {word.text}")  # Add event
            # 'spacing' events are implicitly handled by adding spaces between words

        if line_buffer:  # Append any remaining buffer
            formatted_text_parts.append(line_buffer.strip())

        text_to_save = "\n".join(formatted_text_parts)
    else:
        # Fallback to raw text if not diarizing or no word data
        text_to_save = result.text

    # Create text output directory if it doesn't exist
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
                audio_events.append(word)
            elif word.type == "word":
                print(
                    f"- Word: '{word.text}' (Speaker {word.speaker_id}) at {word.start}s to {word.end}s"
                )
                speakers.add(word.speaker_id)

        ensure_dir_exists(metadata_file_path)
        with open(metadata_file_path, "w", encoding="utf-8") as f:
            f.write(f"Total speakers detected: {len(speakers)}\n")
            for speaker in speakers:
                f.write(f"- {speaker}\n")
            f.write(f"\nTotal audio events detected: {len(audio_events)}\n")
            for event in audio_events:
                f.write(f"- {event.text} at {event.start}s to {event.end}s\n")
        print(f"Metadata saved to {metadata_file_path}")

        print(f"\nTotal speakers detected: {len(speakers)}")
        for speaker in speakers:
            print(f"- {speaker}")
