import os
import json
import argparse
from dotenv import load_dotenv
import io
from pydub import AudioSegment
import openai

from elevenlabs import ElevenLabs

# Korean NLP for paralinguistic analysis
from konlpy.tag import Okt

okt = Okt()

# Load environment variables
load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


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
        timestamps_granularity="word",
    )

    return result


def forced_alignment(audio_path, transcript_text):
    print("Performing forced alignment...")

    try:
        audio = AudioSegment.from_file(audio_path)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)

        alignment_result = client.forced_alignment.create(
            file=buffer, text=transcript_text
        )

        return alignment_result

    except Exception as e:
        print(f"Error during forced alignment: {e}")
        return None


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


def segment_audio_by_turns(audio_path, turns, output_dir):
    print(f"\n=== SEGMENTING AUDIO BY TURNS ===")

    audio = AudioSegment.from_file(audio_path)
    turn_audio_dir = f"{output_dir}/turn_audio"
    ensure_dir_exists(f"{turn_audio_dir}/dummy.wav")

    turn_info = []

    for i, turn in enumerate(turns):
        start_ms = int(turn["start_time"] * 1000)
        end_ms = int(turn["end_time"] * 1000)

        turn_audio = audio[start_ms:end_ms]

        output_file = f"{turn_audio_dir}/{i+1:03d}.wav"
        turn_audio.export(output_file, format="wav")

        turn_data = {
            "turn_id": i + 1,
            "speaker": turn["speaker"],
            "start_time": turn["start_time"],
            "end_time": turn["end_time"],
            "duration": turn["end_time"] - turn["start_time"],
            "text": turn["text"],
            "word_count": turn["word_count"],
            "audio_file": output_file,
        }

        turn_info.append(turn_data)

        print(
            f"Turn {i+1:03d}: {turn['speaker']} ({turn['start_time']:.2f}s - {turn['end_time']:.2f}s) -> {output_file}"
        )

    turns_file = f"{output_dir}/turns_info.json"
    with open(turns_file, "w", encoding="utf-8") as f:
        json.dump(turn_info, f, indent=2, ensure_ascii=False)

    print(f"Turn information saved to: {turns_file}")
    print(f"Total turns created: {len(turns)}")

    return turn_info


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

    # Sort by start time
    audio_events.sort(key=lambda x: x["start_time"])

    return audio_events


def filter_paralinguistic_events(audio_events):
    """
    Filter paralinguistic events using GPT-4o mini with caching system
    Each unique audio tag is classified only once via LLM, then cached for future use

    Required dependencies:
    - openai

    Install with: pip install openai
    Set OPENAI_API_KEY in environment or .env file
    """
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def load_classification_cache(cache_file="audio_tag_cache.json"):
        """Load existing classification cache from JSON file"""
        default_cache = {"paralinguistic": {}, "non_paralinguistic": {}}

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Check if it's the new format
                if "paralinguistic" in data and "non_paralinguistic" in data:
                    return data

                # Migration from old format
                print("Migrating cache from old format to new format...")
                migrated_cache = {"paralinguistic": {}, "non_paralinguistic": {}}

                for tag, info in data.items():
                    if isinstance(info, dict) and "classification" in info:
                        classification = info["classification"]
                        count = info.get("count", 1)
                        migrated_cache[classification][tag] = count

                # Save migrated format
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(migrated_cache, f, indent=2, ensure_ascii=False)

                print(f"Cache migration completed. {len(data)} tags migrated.")
                return migrated_cache

            except Exception as e:
                print(f"Error loading cache file: {e}")
                return default_cache

        return default_cache

    def save_classification_cache(cache, cache_file="audio_tag_cache.json"):
        """Save classification cache to JSON file"""
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving cache file: {e}")

    def classify_with_llm(tag_name):
        """Classify audio tag using GPT-4o mini"""
        prompt = f"""Classify the following audio tag as either paralinguistic or non-paralinguistic sound.

Definitions:
- Paralinguistic: Non-verbal human sounds (laughter, coughs, sighs, interjections)
- Non-paralinguistic: Speech, music, sound effects, mechanical/environmental sounds

Tag to classify: {tag_name}

Response: Either "paralinguistic" or "non_paralinguistic"
"""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
            )

            result = response.choices[0].message.content.strip().lower()

            if "paralinguistic" in result and "non_paralinguistic" not in result:
                return "paralinguistic"
            elif "non_paralinguistic" in result:
                return "non_paralinguistic"
            else:
                print(f"Unexpected LLM response for '{tag_name}': {result}")
                return "non_paralinguistic"  # default fallback

        except Exception as e:
            print(f"Error calling LLM for '{tag_name}': {e}")
            return "non_paralinguistic"  # default fallback

    def get_classification(tag_name, cache):
        """Get classification for tag (from cache or LLM)"""
        clean_tag = (
            tag_name.replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
        )

        if not clean_tag:
            return "non_paralinguistic", "empty_tag"

        # Check cache first
        for classification in ["paralinguistic", "non_paralinguistic"]:
            if clean_tag in cache[classification]:
                cache[classification][clean_tag] += 1
                return classification, "cached"

        # New tag - classify with LLM
        print(f"Classifying new tag with LLM: '{clean_tag}'")
        classification = classify_with_llm(clean_tag)

        # Add to cache
        cache[classification][clean_tag] = 1

        return classification, "llm_classified"

    # Load existing cache
    cache = load_classification_cache()
    cache_file = "audio_tag_cache.json"

    paralinguistic_events = []
    non_paralinguistic_events = []

    print(f"Starting classification with {len(cache)} cached tags")

    for event in audio_events:
        event_text = event["text"].strip()

        classification, reason = get_classification(event_text, cache)
        is_paralinguistic = classification == "paralinguistic"

        event_with_analysis = event.copy()
        event_with_analysis["paralinguistic_score"] = 1.0 if is_paralinguistic else 0.0
        event_with_analysis["classification_reason"] = reason
        event_with_analysis["is_paralinguistic"] = is_paralinguistic
        event_with_analysis["llm_classification"] = classification

        if is_paralinguistic:
            paralinguistic_events.append(event_with_analysis)
        else:
            non_paralinguistic_events.append(event_with_analysis)

    # Save updated cache
    save_classification_cache(cache, cache_file)

    # Print classification summary
    print(f"\nClassification completed:")
    total_unique_tags = len(cache["paralinguistic"]) + len(cache["non_paralinguistic"])
    print(f"- Total unique tags in cache: {total_unique_tags}")
    print(f"- Paralinguistic events: {len(paralinguistic_events)}")
    print(f"- Non-paralinguistic events: {len(non_paralinguistic_events)}")

    if total_unique_tags > 0:
        print("\nTag frequency summary:")

        # Combine all tags from both categories for sorting
        all_tags = []
        for tag, count in cache["paralinguistic"].items():
            all_tags.append((tag, "paralinguistic", count))
        for tag, count in cache["non_paralinguistic"].items():
            all_tags.append((tag, "non_paralinguistic", count))

        # Sort by count (descending)
        sorted_tags = sorted(all_tags, key=lambda x: x[2], reverse=True)

        for tag, classification, count in sorted_tags[:10]:  # Show top 10
            print(f"  {tag}: {classification} ({count} times)")
        if len(sorted_tags) > 10:
            print(f"  ... and {len(sorted_tags) - 10} more tags")

    return paralinguistic_events, non_paralinguistic_events


def segment_audio_by_events(audio_path, audio_events, output_dir):
    if not audio_events:
        print("No audio events to segment.")
        return []

    print(f"\n=== SEGMENTING AUDIO BY EVENTS ===")

    audio = AudioSegment.from_file(audio_path)

    paralinguistic_dir = f"{output_dir}/event_audio/paralinguistic"
    non_paralinguistic_dir = f"{output_dir}/event_audio/non_paralinguistic"
    ensure_dir_exists(f"{paralinguistic_dir}/dummy.wav")
    ensure_dir_exists(f"{non_paralinguistic_dir}/dummy.wav")

    event_info = []
    para_count = 0
    non_para_count = 0

    for i, event in enumerate(audio_events):
        start_ms = int(event["start_time"] * 1000)
        end_ms = int(event["end_time"] * 1000)

        event_audio = audio[start_ms:end_ms]

        is_paralinguistic = event.get("is_paralinguistic", False)
        classification_reason = event.get("classification_reason", "unknown")

        if is_paralinguistic:
            para_count += 1
            output_file = f"{paralinguistic_dir}/para_{para_count:03d}.wav"
            event_type = "paralinguistic"
        else:
            non_para_count += 1
            output_file = f"{non_paralinguistic_dir}/non_para_{non_para_count:03d}.wav"
            event_type = "non_paralinguistic"

        event_audio.export(output_file, format="wav")

        event_data = {
            "event_id": i + 1,
            "text": event["text"],
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "duration": event["duration"],
            "type": event.get("type", "audio_event"),
            "classification": event_type,
            "is_paralinguistic": bool(is_paralinguistic),
            "classification_reason": classification_reason,
            "paralinguistic_score": float(event.get("paralinguistic_score", 0.0)),
            "audio_file": output_file,
        }

        event_info.append(event_data)

        print(
            f"Event {i+1:03d}: [{event_type}] {event['text']} ({event['start_time']:.2f}s - {event['end_time']:.2f}s) -> {output_file}"
        )

    events_audio_info_file = f"{output_dir}/events_audio_info.json"
    with open(events_audio_info_file, "w", encoding="utf-8") as f:
        json.dump(event_info, f, indent=2, ensure_ascii=False)

    print(f"Event audio information saved to: {events_audio_info_file}")
    print(f"Total event audio files created: {len(audio_events)}")
    print(f"- Paralinguistic events: {para_count} files in {paralinguistic_dir}")
    print(
        f"- Non-paralinguistic events: {non_para_count} files in {non_paralinguistic_dir}"
    )

    return event_info


def save_audio_events(audio_events, output_dir, audio_path=None):
    if not audio_events:
        print("No audio events detected.")
        return None

    print(f"\n=== PROCESSING AUDIO EVENTS ===")

    # Filter paralinguistic events
    paralinguistic_events, non_paralinguistic_events = filter_paralinguistic_events(
        audio_events
    )

    # Save all audio events as JSON
    audio_events_file = f"{output_dir}/audio_events.json"
    with open(audio_events_file, "w", encoding="utf-8") as f:
        json.dump(audio_events, f, indent=2, ensure_ascii=False)

    # Save paralinguistic events separately
    paralinguistic_file = f"{output_dir}/paralinguistic_events.json"
    with open(paralinguistic_file, "w", encoding="utf-8") as f:
        json.dump(paralinguistic_events, f, indent=2, ensure_ascii=False)

    # Save non-paralinguistic events separately
    non_paralinguistic_file = f"{output_dir}/non_paralinguistic_events.json"
    with open(non_paralinguistic_file, "w", encoding="utf-8") as f:
        json.dump(non_paralinguistic_events, f, indent=2, ensure_ascii=False)

    # Save audio events as readable text
    audio_events_text_file = f"{output_dir}/audio_events.txt"
    with open(audio_events_text_file, "w", encoding="utf-8") as f:
        f.write("ALL AUDIO EVENTS TIMELINE\n")
        f.write("=" * 50 + "\n\n")

        for i, event in enumerate(audio_events, 1):
            f.write(
                f"{i:03d}. [{event['start_time']:.2f}s - {event['end_time']:.2f}s] "
                f"({event['duration']:.2f}s) {event['text']}\n"
            )

    # Save paralinguistic events as readable text
    paralinguistic_text_file = f"{output_dir}/paralinguistic_events.txt"
    with open(paralinguistic_text_file, "w", encoding="utf-8") as f:
        f.write("PARALINGUISTIC EVENTS TIMELINE\n")
        f.write("=" * 50 + "\n")
        f.write("(Human speech-related non-verbal sounds)\n\n")

        if paralinguistic_events:
            for i, event in enumerate(paralinguistic_events, 1):
                f.write(
                    f"{i:03d}. [{event['start_time']:.2f}s - {event['end_time']:.2f}s] "
                    f"({event['duration']:.2f}s) {event['text']}\n"
                )
        else:
            f.write("No paralinguistic events detected.\n")

    # Save non-paralinguistic events as readable text
    non_paralinguistic_text_file = f"{output_dir}/non_paralinguistic_events.txt"
    with open(non_paralinguistic_text_file, "w", encoding="utf-8") as f:
        f.write("NON-PARALINGUISTIC EVENTS TIMELINE\n")
        f.write("=" * 50 + "\n")
        f.write("(Environmental sounds, music, etc.)\n\n")

        if non_paralinguistic_events:
            for i, event in enumerate(non_paralinguistic_events, 1):
                f.write(
                    f"{i:03d}. [{event['start_time']:.2f}s - {event['end_time']:.2f}s] "
                    f"({event['duration']:.2f}s) {event['text']}\n"
                )
        else:
            f.write("No non-paralinguistic events detected.\n")

    # Combine all classified events for audio segmentation
    all_classified_events = paralinguistic_events + non_paralinguistic_events
    all_classified_events.sort(key=lambda x: x["start_time"])

    # Segment audio by events if audio path is provided
    event_audio_info = None
    if audio_path and os.path.exists(audio_path):
        event_audio_info = segment_audio_by_events(
            audio_path, all_classified_events, output_dir
        )

    print(f"Audio events saved to:")
    print(f"- All events JSON: {audio_events_file}")
    print(f"- All events text: {audio_events_text_file}")
    print(f"- Paralinguistic JSON: {paralinguistic_file}")
    print(f"- Paralinguistic text: {paralinguistic_text_file}")
    print(f"- Non-paralinguistic JSON: {non_paralinguistic_file}")
    print(f"- Non-paralinguistic text: {non_paralinguistic_text_file}")
    if event_audio_info:
        print(f"- Event audio directory: {output_dir}/event_audio")
        print(f"- Event audio info: {output_dir}/events_audio_info.json")

    print(f"\nEvent statistics:")
    print(f"- Total audio events: {len(audio_events)}")
    print(f"- Paralinguistic events: {len(paralinguistic_events)}")
    print(f"- Non-paralinguistic events: {len(non_paralinguistic_events)}")
    if event_audio_info:
        print(f"- Event audio files created: {len(event_audio_info)}")

    # Print paralinguistic event types
    if paralinguistic_events:
        para_event_types = {}
        for event in paralinguistic_events:
            event_text = event["text"].lower()
            if event_text in para_event_types:
                para_event_types[event_text] += 1
            else:
                para_event_types[event_text] = 1

        print(f"\nParalinguistic event types detected:")
        for event_type, count in sorted(para_event_types.items()):
            print(f"- {event_type}: {count} times")

    return {
        "events_file": audio_events_file,
        "events_text_file": audio_events_text_file,
        "paralinguistic_file": paralinguistic_file,
        "paralinguistic_text_file": paralinguistic_text_file,
        "non_paralinguistic_file": non_paralinguistic_file,
        "non_paralinguistic_text_file": non_paralinguistic_text_file,
        "event_audio_dir": f"{output_dir}/event_audio" if event_audio_info else None,
        "events_audio_info_file": f"{output_dir}/events_audio_info.json"
        if event_audio_info
        else None,
        "total_events": len(audio_events),
        "paralinguistic_count": len(paralinguistic_events),
        "non_paralinguistic_count": len(non_paralinguistic_events),
    }


def process_full_audio_analysis(audio_path):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    output_dir = f"elevenlabs_results/{base_name}"
    ensure_dir_exists(f"{output_dir}/dummy.txt")

    transcription_file = f"{output_dir}/transcription.json"
    diarized_file = f"{output_dir}/diarized.txt"
    alignment_file = f"{output_dir}/alignment.json"
    summary_file = f"{output_dir}/summary.json"

    result = None
    transcript_text = None

    if os.path.exists(transcription_file) and os.path.exists(diarized_file):
        print("=== LOADING EXISTING TRANSCRIPTION ===")
        print(f"Found existing files:")
        print(f"- {transcription_file}")
        print(f"- {diarized_file}")

        with open(transcription_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            transcript_text = existing_data.get("text", "")

        print("Skipping speech-to-text processing...")
    else:
        print("=== SPEECH-TO-TEXT WITH DIARIZATION ===")
        result = speech_to_text_with_diarization(audio_path)
        transcript_text = result.text

        with open(transcription_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "language_code": result.language_code,
                    "language_probability": result.language_probability,
                    "text": result.text,
                    "words": [
                        {
                            "text": word.text,
                            "type": word.type,
                            "start": word.start,
                            "end": word.end,
                            "speaker_id": getattr(word, "speaker_id", None),
                            "logprob": getattr(word, "logprob", None),
                        }
                        for word in result.words
                    ]
                    if hasattr(result, "words")
                    else [],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Transcription saved to: {transcription_file}")

        diarized_text_parts = []
        current_speaker = None
        line_buffer = ""
        speakers = set()
        audio_events = []

        if hasattr(result, "words") and result.words:
            for word in result.words:
                if word.type == "word":
                    speaker = getattr(word, "speaker_id", "Unknown")
                    speakers.add(speaker)

                    if word.text:
                        if speaker != current_speaker:
                            if line_buffer:
                                diarized_text_parts.append(line_buffer.strip())
                            line_buffer = f"{speaker}: {word.text}"
                            current_speaker = speaker
                        else:
                            if line_buffer:
                                line_buffer += f" {word.text}"
                            else:
                                line_buffer = f"{speaker}: {word.text}"
                elif word.type == "audio_event":
                    if line_buffer:
                        diarized_text_parts.append(line_buffer.strip())
                        line_buffer = ""
                        current_speaker = None
                    audio_events.append(
                        {
                            "text": word.text,
                            "start": word.start,
                            "end": word.end,
                            "type": "audio_event",
                        }
                    )
                    diarized_text_parts.append(f"[AUDIO EVENT: {word.text}]")

            if line_buffer:
                diarized_text_parts.append(line_buffer.strip())

        diarized_text = "\n".join(diarized_text_parts)

        with open(diarized_file, "w", encoding="utf-8") as f:
            f.write(diarized_text)

        print(f"Diarized text saved to: {diarized_file}")

    if os.path.exists(alignment_file):
        print("\n=== LOADING EXISTING FORCED ALIGNMENT ===")
        print(f"Found existing alignment file: {alignment_file}")
        print("Skipping forced alignment processing...")
    else:
        print("\n=== FORCED ALIGNMENT ===")
        if transcript_text:
            alignment_result = forced_alignment(audio_path, transcript_text)

            if alignment_result:
                alignment_data = {
                    "characters": [
                        {"text": char.text, "start": char.start, "end": char.end}
                        for char in alignment_result.characters
                    ]
                    if hasattr(alignment_result, "characters")
                    else [],
                    "words": [
                        {"text": word.text, "start": word.start, "end": word.end}
                        for word in alignment_result.words
                    ]
                    if hasattr(alignment_result, "words")
                    else [],
                }

                with open(alignment_file, "w", encoding="utf-8") as f:
                    json.dump(alignment_data, f, indent=2, ensure_ascii=False)
                print(f"Forced alignment saved to: {alignment_file}")
        else:
            print("No transcript text available for forced alignment")

    turns_info_file = f"{output_dir}/turns_info.json"
    if os.path.exists(turns_info_file):
        print("\n=== LOADING EXISTING TURN SEGMENTATION ===")
        print(f"Found existing turn segmentation: {turns_info_file}")
        print("Skipping audio segmentation...")

        with open(turns_info_file, "r", encoding="utf-8") as f:
            turn_info = json.load(f)
    else:
        print("\n=== TURN SEGMENTATION ===")
        with open(transcription_file, "r", encoding="utf-8") as f:
            transcription_data = json.load(f)

        if transcription_data.get("words"):
            turns = extract_turns_from_words(transcription_data["words"])
            turn_info = segment_audio_by_turns(audio_path, turns, output_dir)
        else:
            print("No word-level data available for turn segmentation")
            turn_info = []

    # Audio Events Processing
    audio_events_file = f"{output_dir}/audio_events.json"
    audio_events_info = None

    if os.path.exists(audio_events_file):
        print("\n=== LOADING EXISTING AUDIO EVENTS ===")
        print(f"Found existing audio events file: {audio_events_file}")
        print("Skipping audio events processing...")

        with open(audio_events_file, "r", encoding="utf-8") as f:
            audio_events_data = json.load(f)
            audio_events_info = {
                "events_file": audio_events_file,
                "events_text_file": f"{output_dir}/audio_events.txt",
                "paralinguistic_file": f"{output_dir}/paralinguistic_events.json",
                "paralinguistic_text_file": f"{output_dir}/paralinguistic_events.txt",
                "non_paralinguistic_file": f"{output_dir}/non_paralinguistic_events.json",
                "non_paralinguistic_text_file": f"{output_dir}/non_paralinguistic_events.txt",
                "event_audio_dir": f"{output_dir}/event_audio",
                "events_audio_info_file": f"{output_dir}/events_audio_info.json",
                "total_events": len(audio_events_data),
                "event_types": {},
            }
    else:
        with open(transcription_file, "r", encoding="utf-8") as f:
            transcription_data = json.load(f)

        if transcription_data.get("words"):
            audio_events = extract_audio_events(transcription_data["words"])
            audio_events_info = save_audio_events(audio_events, output_dir, audio_path)
        else:
            print("No word-level data available for audio events processing")

    if result:
        speakers = set()
        audio_events = []

        if hasattr(result, "words") and result.words:
            for word in result.words:
                if word.type == "word":
                    speaker = getattr(word, "speaker_id", "Unknown")
                    speakers.add(speaker)
                elif word.type == "audio_event":
                    audio_events.append(
                        {
                            "text": word.text,
                            "start": word.start,
                            "end": word.end,
                            "type": "audio_event",
                        }
                    )

        summary = {
            "audio_file": audio_path,
            "language": result.language_code,
            "language_confidence": result.language_probability,
            "total_speakers": len(speakers),
            "speakers": list(speakers),
            "audio_events_count": len(audio_events),
            "audio_events": audio_events,
            "transcription_length": len(result.text),
            "word_count": len([w for w in result.words if w.type == "word"])
            if hasattr(result, "words")
            else 0,
            "total_turns": len(turn_info),
            "files_generated": {
                "transcription": transcription_file,
                "diarized": diarized_file,
                "alignment": alignment_file if os.path.exists(alignment_file) else None,
                "turns_info": turns_info_file,
                "turn_audio_dir": f"{output_dir}/turn_audio" if turn_info else None,
                "audio_events": audio_events_info["events_file"]
                if audio_events_info
                else None,
                "audio_events_text": audio_events_info["events_text_file"]
                if audio_events_info
                else None,
                "paralinguistic_events": audio_events_info["paralinguistic_file"]
                if audio_events_info
                else None,
                "paralinguistic_events_text": audio_events_info[
                    "paralinguistic_text_file"
                ]
                if audio_events_info
                else None,
                "non_paralinguistic_events": audio_events_info[
                    "non_paralinguistic_file"
                ]
                if audio_events_info
                else None,
                "non_paralinguistic_events_text": audio_events_info[
                    "non_paralinguistic_text_file"
                ]
                if audio_events_info
                else None,
                "event_audio_dir": audio_events_info["event_audio_dir"]
                if audio_events_info
                else None,
                "events_audio_info": audio_events_info["events_audio_info_file"]
                if audio_events_info
                else None,
            },
        }

        print(f"\n=== SUMMARY ===")
        print(
            f"Language: {result.language_code} (confidence: {result.language_probability:.2f})"
        )
        print(f"Total speakers: {len(speakers)}")
        print(f"Speakers: {', '.join(speakers)}")
        print(f"Audio events detected: {len(audio_events)}")
        print(f"Total turns: {len(turn_info)}")
        if audio_events_info:
            print(
                f"Audio events file generated: {audio_events_info['total_events']} events"
            )
    else:
        with open(transcription_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)

        summary = {
            "audio_file": audio_path,
            "language": existing_data.get("language_code", "unknown"),
            "language_confidence": existing_data.get("language_probability", 0),
            "total_speakers": len(
                set(
                    [
                        w.get("speaker_id")
                        for w in existing_data.get("words", [])
                        if w.get("speaker_id")
                    ]
                )
            ),
            "speakers": list(
                set(
                    [
                        w.get("speaker_id")
                        for w in existing_data.get("words", [])
                        if w.get("speaker_id")
                    ]
                )
            ),
            "audio_events_count": len(
                [
                    w
                    for w in existing_data.get("words", [])
                    if w.get("type") == "audio_event"
                ]
            ),
            "audio_events": [
                w
                for w in existing_data.get("words", [])
                if w.get("type") == "audio_event"
            ],
            "transcription_length": len(existing_data.get("text", "")),
            "word_count": len(
                [w for w in existing_data.get("words", []) if w.get("type") == "word"]
            ),
            "total_turns": len(turn_info),
            "files_generated": {
                "transcription": transcription_file,
                "diarized": diarized_file,
                "alignment": alignment_file if os.path.exists(alignment_file) else None,
                "turns_info": turns_info_file,
                "turn_audio_dir": f"{output_dir}/turn_audio" if turn_info else None,
                "audio_events": audio_events_info["events_file"]
                if audio_events_info
                else None,
                "audio_events_text": audio_events_info["events_text_file"]
                if audio_events_info
                else None,
                "paralinguistic_events": audio_events_info["paralinguistic_file"]
                if audio_events_info
                else None,
                "paralinguistic_events_text": audio_events_info[
                    "paralinguistic_text_file"
                ]
                if audio_events_info
                else None,
                "non_paralinguistic_events": audio_events_info[
                    "non_paralinguistic_file"
                ]
                if audio_events_info
                else None,
                "non_paralinguistic_events_text": audio_events_info[
                    "non_paralinguistic_text_file"
                ]
                if audio_events_info
                else None,
                "event_audio_dir": audio_events_info["event_audio_dir"]
                if audio_events_info
                else None,
                "events_audio_info": audio_events_info["events_audio_info_file"]
                if audio_events_info
                else None,
            },
        }

        print(f"\n=== SUMMARY (FROM EXISTING FILES) ===")
        print(
            f"Language: {summary['language']} (confidence: {summary['language_confidence']:.2f})"
        )
        print(f"Total speakers: {summary['total_speakers']}")
        print(f"Speakers: {', '.join(summary['speakers'])}")
        print(f"Audio events detected: {summary['audio_events_count']}")
        print(f"Total turns: {len(turn_info)}")
        if audio_events_info:
            print(
                f"Audio events file generated: {audio_events_info['total_events']} events"
            )

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Summary saved to: {summary_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process audio with ElevenLabs full features"
    )
    parser.add_argument("audio_path", help="Path to the audio file")

    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}")
        exit(1)

    process_full_audio_analysis(args.audio_path)
