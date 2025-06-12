import os
import json
import argparse
from dotenv import load_dotenv
import io
from pydub import AudioSegment
import openai
from elevenlabs import ElevenLabs
from konlpy.tag import Okt

okt = Okt()
load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"), timeout=1000)


def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    return file_path


def speech_to_text_with_diarization(audio_path):
    print(f"Processing audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    print("Converting speech to text with diarization and audio tag detection...")
    result = client.speech_to_text.convert(
        file=buffer,
        model_id="scribe_v1_experimental",
        diarize=True,
        tag_audio_events=True,
    )
    return result


def extract_turns_from_words(words):
    turns = []
    current_speaker = None
    current_turn_words = []
    for word in words:
        if word.get("type") == "word" and word.get("text"):
            speaker = word.get("speaker_id", "Unknown")
            if speaker != current_speaker:
                if current_turn_words:
                    turn_start = current_turn_words[0]["start"]
                    turn_end = current_turn_words[-1]["end"]
                    turn_text = " ".join([w["text"] for w in current_turn_words])
                    turns.append(
                        {
                            "speaker": current_speaker,
                            "start_time": turn_start,
                            "end_time": turn_end,
                            "text": turn_text,
                            "word_count": len(current_turn_words),
                        }
                    )
                current_speaker = speaker
                current_turn_words = [word]
            else:
                current_turn_words.append(word)
    if current_turn_words:
        turn_start = current_turn_words[0]["start"]
        turn_end = current_turn_words[-1]["end"]
        turn_text = " ".join([w["text"] for w in current_turn_words])
        turns.append(
            {
                "speaker": current_speaker,
                "start_time": turn_start,
                "end_time": turn_end,
                "text": turn_text,
                "word_count": len(current_turn_words),
            }
        )
    return turns


def extract_audio_events(words):
    audio_events = []
    for word in words:
        if word.get("type") == "audio_event":
            audio_events.append(
                {
                    "text": word.get("text", ""),
                    "start_time": word.get("start", 0),
                    "end_time": word.get("end", 0),
                    "duration": word.get("end", 0) - word.get("start", 0),
                    "type": "audio_event",
                }
            )
    audio_events.sort(key=lambda x: x["start_time"])
    return audio_events


def filter_paralinguistic_events(audio_events):
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def load_classification_cache(cache_file="audio_tag_cache.json"):
        default_cache = {"paralinguistic": {}, "non_paralinguistic": {}}
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
                if "paralinguistic" not in cache:
                    cache["paralinguistic"] = {}
                if "non_paralinguistic" not in cache:
                    cache["non_paralinguistic"] = {}
                return cache
        return default_cache

    def save_classification_cache(cache, cache_file="audio_tag_cache.json"):
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    def classify_with_llm(tag_name):
        prompt = f"""Classify the following audio tag as either paralinguistic or non-paralinguistic sound.
Definitions:
- Paralinguistic: Non-verbal human sounds (laughter, coughs, sighs, interjections)
- Non-paralinguistic: Speech, music, sound effects, mechanical/environmental sounds

Tag to classify: {tag_name}

Response: Either 'paralinguistic' or 'non_paralinguistic'"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        return response.choices[0].message.content.strip().lower()

    def get_classification(tag_name, cache):
        if tag_name in cache["paralinguistic"]:
            cache["paralinguistic"][tag_name] += 1
            return "paralinguistic", cache["paralinguistic"][tag_name]
        elif tag_name in cache["non_paralinguistic"]:
            cache["non_paralinguistic"][tag_name] += 1
            return "non_paralinguistic", cache["non_paralinguistic"][tag_name]
        else:
            classification = classify_with_llm(tag_name)
            count = 1
            if classification == "paralinguistic":
                cache["paralinguistic"][tag_name] = count
            else:
                cache["non_paralinguistic"][tag_name] = count
            print(f"LLM classified '{tag_name}' as {classification}")
            return classification, count

    cache = load_classification_cache()
    paralinguistic_events = []
    non_paralinguistic_events = []

    unique_tags = list(set([event["text"] for event in audio_events]))
    print(f"Found {len(unique_tags)} unique audio tags: {unique_tags}")

    for event in audio_events:
        tag_name = event["text"]
        classification, count = get_classification(tag_name, cache)
        event_with_score = event.copy()
        event_with_score["classification"] = classification
        event_with_score["occurrence_count"] = count

        if classification == "paralinguistic":
            paralinguistic_events.append(event_with_score)
        else:
            non_paralinguistic_events.append(event_with_score)

    save_classification_cache(cache)

    print(f"Classification complete:")
    print(f"- Paralinguistic events: {len(paralinguistic_events)}")
    print(f"- Non-paralinguistic events: {len(non_paralinguistic_events)}")

    return paralinguistic_events, non_paralinguistic_events


def generate_diarized_text(words):
    diarized_lines = []
    current_speaker = None
    current_line = ""

    for word in words:
        if word.get("type") == "word" and word.get("text"):
            speaker = word.get("speaker_id", "Unknown")
            text = word["text"]

            if speaker != current_speaker:
                if current_line:
                    diarized_lines.append(current_line)
                current_line = f"{speaker}: {text}"
                current_speaker = speaker
            else:
                current_line += f" {text}"
        elif word.get("type") == "audio_event":
            event_text = f"[{word.get('text', 'audio_event')}]"
            if current_line:
                current_line += f" {event_text}"
            else:
                current_line = event_text

    if current_line:
        diarized_lines.append(current_line)

    return "\n".join(diarized_lines)


def process_full_audio_analysis_efficient(audio_path):
    print(f"\n=== ASR-BASED AUDIO ANALYSIS ===")
    print(f"Audio file: {audio_path}")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = "elevenlabs_results"
    ensure_dir_exists(f"{output_dir}/dummy.txt")

    unified_output_file = f"{output_dir}/{base_name}.json"

    if os.path.exists(unified_output_file):
        print(f"Unified analysis already exists: {unified_output_file}")
        with open(unified_output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    result = speech_to_text_with_diarization(audio_path)

    if not result or not hasattr(result, "words") or not result.words:
        print("No transcription result or words found")
        return None

    words = []
    speakers = set()

    for word in result.words:
        word_dict = {
            "text": word.text,
            "type": word.type,
            "start": word.start,
            "end": word.end,
            "timestamp_source": "asr_original",
        }

        if hasattr(word, "speaker_id"):
            word_dict["speaker_id"] = word.speaker_id
            if word.type == "word":
                speakers.add(word.speaker_id)

        words.append(word_dict)

    words.sort(key=lambda x: x.get("start", 0))

    turns = extract_turns_from_words(words)
    audio_events = extract_audio_events(words)

    paralinguistic_events = []
    non_paralinguistic_events = []

    if audio_events:
        paralinguistic_events, non_paralinguistic_events = filter_paralinguistic_events(
            audio_events
        )

    diarized_text = generate_diarized_text(words)

    unified_data = {
        "metadata": {
            "audio_file": audio_path,
            "audio_file_basename": base_name,
            "language": result.language_code,
            "language_confidence": result.language_probability,
            "processing_timestamp": __import__("datetime").datetime.now().isoformat(),
            "timestamp_source": "asr_original",
        },
        "speakers": {
            "total_count": len(speakers),
            "speaker_list": sorted(list(speakers)),
        },
        "transcription": {
            "full_text": result.text,
            "word_count": len([w for w in words if w["type"] == "word"]),
            "total_duration": max([w["end"] for w in words if w.get("end")])
            if words
            else 0,
            "words": words,
        },
        "diarization": {
            "diarized_text": diarized_text,
            "total_turns": len(turns),
            "turns": [
                {
                    "turn_id": i + 1,
                    "speaker": turn["speaker"],
                    "start_time": turn["start_time"],
                    "end_time": turn["end_time"],
                    "duration": turn["end_time"] - turn["start_time"],
                    "text": turn["text"],
                    "word_count": turn["word_count"],
                    "timestamp_source": "asr_original",
                }
                for i, turn in enumerate(turns)
            ],
        },
        "audio_events": {
            "total_count": len(audio_events),
            "paralinguistic_count": len(paralinguistic_events),
            "non_paralinguistic_count": len(non_paralinguistic_events),
            "paralinguistic_events": [
                {
                    "event_id": i + 1,
                    "text": event["text"],
                    "start_time": event["start_time"],
                    "end_time": event["end_time"],
                    "duration": event["duration"],
                    "classification": event["classification"],
                    "occurrence_count": event["occurrence_count"],
                    "timestamp_source": "asr_original",
                }
                for i, event in enumerate(paralinguistic_events)
            ],
            "non_paralinguistic_events": [
                {
                    "event_id": i + len(paralinguistic_events) + 1,
                    "text": event["text"],
                    "start_time": event["start_time"],
                    "end_time": event["end_time"],
                    "duration": event["duration"],
                    "classification": event["classification"],
                    "occurrence_count": event["occurrence_count"],
                    "timestamp_source": "asr_original",
                }
                for i, event in enumerate(non_paralinguistic_events)
            ],
            "all_events": audio_events,
        },
    }

    with open(unified_output_file, "w", encoding="utf-8") as f:
        json.dump(unified_data, f, indent=2, ensure_ascii=False)

    print(f"\n=== ANALYSIS COMPLETE ===")
    print(
        f"Language: {result.language_code} (confidence: {result.language_probability:.2f})"
    )
    print(f"Total speakers: {len(speakers)}")
    print(f"Speakers: {', '.join(sorted(speakers))}")
    print(f"Total turns: {len(turns)}")
    print(f"Total audio events: {len(audio_events)}")
    print(f"- Paralinguistic: {len(paralinguistic_events)}")
    print(f"- Non-paralinguistic: {len(non_paralinguistic_events)}")
    print(f"Unified analysis saved to: {unified_output_file}")

    return unified_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASR-based audio analysis with unified output"
    )
    parser.add_argument("audio_path", help="Path to the audio file")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}")
        exit(1)

    process_full_audio_analysis_efficient(args.audio_path)
