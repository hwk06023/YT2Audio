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
    )
    return result


def forced_alignment(audio_path, transcript_text):
    print("Performing forced alignment...")
    audio = AudioSegment.from_file(audio_path)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    alignment_result = client.forced_alignment.create(file=buffer, text=transcript_text)
    return alignment_result


def extract_turns_from_words(words, method_name="unknown"):
    """Extract turns from words with method tracking"""
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
                            "method": method_name,
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
                "method": method_name,
            }
        )

    return turns


def match_words_with_forced_alignment(asr_words, alignment_words):
    """Match ASR words with forced alignment timestamps"""
    if not alignment_words:
        print("No forced alignment data available")
        return asr_words

    # Filter only actual words from forced alignment (exclude audio events, effects, etc.)
    alignment_word_list = []
    for w in alignment_words:
        text = w.get("text", "").strip()
        # Skip audio events, effects, and empty text
        if (
            text
            and not (text.startswith("[") and text.endswith("]"))
            and not text.startswith("(")
            and not text.startswith("<")
        ):
            alignment_word_list.append(w)

    word_events = [w for w in asr_words if w.get("type") == "word"]
    audio_events = [w for w in asr_words if w.get("type") == "audio_event"]

    print(
        f"Matching {len(word_events)} ASR words with {len(alignment_word_list)} forced alignment words"
    )
    print(
        f"Filtered out {len(alignment_words) - len(alignment_word_list)} non-word items from forced alignment"
    )

    updated_words = []
    alignment_idx = 0

    for asr_word in word_events:
        asr_text = asr_word.get("text", "").strip().replace(" ", "").lower()

        accumulated_text = ""
        matched_alignment_words = []
        start_idx = alignment_idx

        while alignment_idx < len(alignment_word_list):
            alignment_word = alignment_word_list[alignment_idx]
            alignment_text = (
                alignment_word.get("text", "").strip().replace(" ", "").lower()
            )

            accumulated_text += alignment_text
            matched_alignment_words.append(alignment_word)
            alignment_idx += 1

            if accumulated_text == asr_text:
                updated_word = asr_word.copy()
                updated_word["start"] = matched_alignment_words[0]["start"]
                updated_word["end"] = matched_alignment_words[-1]["end"]
                updated_word["timestamp_source"] = "forced_alignment"
                updated_word["alignment_word_count"] = len(matched_alignment_words)
                updated_words.append(updated_word)
                break
            elif len(accumulated_text) > len(asr_text):
                raise ValueError(
                    f"Text mismatch: ASR word '{asr_word.get('text')}' vs accumulated '{accumulated_text}'"
                )
        else:
            raise ValueError(
                f"Incomplete match: ASR word '{asr_word.get('text')}' vs accumulated '{accumulated_text}'"
            )

    for audio_event in audio_events:
        audio_event_copy = audio_event.copy()
        audio_event_copy["timestamp_source"] = "asr_original"
        updated_words.append(audio_event_copy)

    updated_words.sort(key=lambda x: x.get("start", 0))
    return updated_words


def analyze_turn_differences(asr_turns, fa_turns):
    """Analyze differences between ASR and forced alignment turns"""
    differences = []

    print(f"\n=== TURN COMPARISON ANALYSIS ===")
    print(f"ASR Turns: {len(asr_turns)}")
    print(f"Forced Alignment Turns: {len(fa_turns)}")

    max_turns = max(len(asr_turns), len(fa_turns))

    for i in range(max_turns):
        asr_turn = asr_turns[i] if i < len(asr_turns) else None
        fa_turn = fa_turns[i] if i < len(fa_turns) else None

        if asr_turn and fa_turn:
            time_diff_start = abs(asr_turn["start_time"] - fa_turn["start_time"])
            time_diff_end = abs(asr_turn["end_time"] - fa_turn["end_time"])
            duration_diff = abs(
                (asr_turn["end_time"] - asr_turn["start_time"])
                - (fa_turn["end_time"] - fa_turn["start_time"])
            )

            diff_data = {
                "turn_index": i + 1,
                "speaker_match": asr_turn["speaker"] == fa_turn["speaker"],
                "text_match": asr_turn["text"] == fa_turn["text"],
                "start_time_diff": time_diff_start,
                "end_time_diff": time_diff_end,
                "duration_diff": duration_diff,
                "asr_duration": asr_turn["end_time"] - asr_turn["start_time"],
                "fa_duration": fa_turn["end_time"] - fa_turn["start_time"],
                "asr_turn": asr_turn,
                "fa_turn": fa_turn,
            }
            differences.append(diff_data)

            if time_diff_start > 0.1 or time_diff_end > 0.1:
                print(f"Turn {i+1}: Significant timing difference")
                print(
                    f"  ASR: {asr_turn['start_time']:.3f}s - {asr_turn['end_time']:.3f}s"
                )
                print(
                    f"  FA:  {fa_turn['start_time']:.3f}s - {fa_turn['end_time']:.3f}s"
                )
                print(f"  Diff: start={time_diff_start:.3f}s, end={time_diff_end:.3f}s")

        elif asr_turn and not fa_turn:
            print(
                f"Turn {i+1}: Only in ASR - {asr_turn['speaker']}: {asr_turn['text'][:50]}..."
            )
        elif fa_turn and not asr_turn:
            print(
                f"Turn {i+1}: Only in FA - {fa_turn['speaker']}: {fa_turn['text'][:50]}..."
            )

    return differences


def generate_comparison_report(differences, output_dir, base_name):
    """Generate detailed comparison report"""

    if not differences:
        print("No turn differences to analyze")
        return

    total_turns = len(differences)
    text_matches = sum(1 for d in differences if d["text_match"])
    speaker_matches = sum(1 for d in differences if d["speaker_match"])

    avg_start_diff = sum(d["start_time_diff"] for d in differences) / total_turns
    avg_end_diff = sum(d["end_time_diff"] for d in differences) / total_turns
    avg_duration_diff = sum(d["duration_diff"] for d in differences) / total_turns

    max_start_diff = max(d["start_time_diff"] for d in differences)
    max_end_diff = max(d["end_time_diff"] for d in differences)
    max_duration_diff = max(d["duration_diff"] for d in differences)

    total_asr_duration = sum(d["asr_duration"] for d in differences)
    total_fa_duration = sum(d["fa_duration"] for d in differences)

    report = {
        "summary": {
            "total_turns_compared": total_turns,
            "text_match_rate": text_matches / total_turns * 100,
            "speaker_match_rate": speaker_matches / total_turns * 100,
            "avg_start_time_diff": avg_start_diff,
            "avg_end_time_diff": avg_end_diff,
            "avg_duration_diff": avg_duration_diff,
            "max_start_time_diff": max_start_diff,
            "max_end_time_diff": max_end_diff,
            "max_duration_diff": max_duration_diff,
            "total_asr_duration": total_asr_duration,
            "total_fa_duration": total_fa_duration,
            "total_duration_diff": abs(total_asr_duration - total_fa_duration),
        },
        "detailed_differences": differences,
    }

    report_file = f"{output_dir}/{base_name}_comparison_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"Total turns compared: {total_turns}")
    print(
        f"Text match rate: {text_matches}/{total_turns} ({text_matches/total_turns*100:.1f}%)"
    )
    print(
        f"Speaker match rate: {speaker_matches}/{total_turns} ({speaker_matches/total_turns*100:.1f}%)"
    )
    print(f"Average timing differences:")
    print(f"  Start time: {avg_start_diff:.3f}s")
    print(f"  End time: {avg_end_diff:.3f}s")
    print(f"  Duration: {avg_duration_diff:.3f}s")
    print(f"Maximum timing differences:")
    print(f"  Start time: {max_start_diff:.3f}s")
    print(f"  End time: {max_end_diff:.3f}s")
    print(f"  Duration: {max_duration_diff:.3f}s")
    print(
        f"Total duration difference: {abs(total_asr_duration - total_fa_duration):.3f}s"
    )
    print(f"Comparison report saved to: {report_file}")

    return report


def segment_audio_by_turns(audio_path, turns, output_dir, method_name):
    """
    Segment audio file based on turn timestamps
    """
    print(f"\n=== SEGMENTING AUDIO BY TURNS ({method_name.upper()}) ===")

    audio = AudioSegment.from_file(audio_path)
    method_dir = f"{output_dir}/audio_segments/{method_name}"
    ensure_dir_exists(f"{method_dir}/dummy.wav")

    segment_info = []

    for i, turn in enumerate(turns):
        start_ms = int(turn["start_time"] * 1000)
        end_ms = int(turn["end_time"] * 1000)

        # Extract audio segment
        turn_audio = audio[start_ms:end_ms]

        # Generate filename
        speaker = turn["speaker"]
        duration = turn["end_time"] - turn["start_time"]
        output_file = f"{method_dir}/{i+1:03d}_{speaker}_{duration:.2f}s.wav"

        # Export audio segment
        turn_audio.export(output_file, format="wav")

        segment_data = {
            "turn_id": i + 1,
            "speaker": speaker,
            "start_time": turn["start_time"],
            "end_time": turn["end_time"],
            "duration": duration,
            "text": turn["text"],
            "word_count": turn["word_count"],
            "method": method_name,
            "audio_file": output_file,
            "file_size_mb": os.path.getsize(output_file) / (1024 * 1024),
        }

        segment_info.append(segment_data)

        print(
            f"Turn {i+1:03d}: {speaker} ({turn['start_time']:.2f}s - {turn['end_time']:.2f}s) -> {output_file}"
        )

    # Save segment info
    segments_file = f"{method_dir}/segments_info.json"
    with open(segments_file, "w", encoding="utf-8") as f:
        json.dump(segment_info, f, indent=2, ensure_ascii=False)

    print(f"Audio segments saved to: {method_dir}")
    print(f"Segment info saved to: {segments_file}")
    print(f"Total segments created: {len(turns)}")

    return segment_info


def create_comparison_audio_report(asr_segments, fa_segments, output_dir, base_name):
    """
    Create a detailed report comparing audio segments from both methods
    """
    comparison_segments = []

    max_segments = max(len(asr_segments), len(fa_segments))

    for i in range(max_segments):
        asr_seg = asr_segments[i] if i < len(asr_segments) else None
        fa_seg = fa_segments[i] if i < len(fa_segments) else None

        if asr_seg and fa_seg:
            duration_diff = abs(asr_seg["duration"] - fa_seg["duration"])
            start_diff = abs(asr_seg["start_time"] - fa_seg["start_time"])
            end_diff = abs(asr_seg["end_time"] - fa_seg["end_time"])

            comparison_data = {
                "segment_index": i + 1,
                "speaker_match": asr_seg["speaker"] == fa_seg["speaker"],
                "text_match": asr_seg["text"] == fa_seg["text"],
                "duration_difference": duration_diff,
                "start_time_difference": start_diff,
                "end_time_difference": end_diff,
                "asr_segment": {
                    "audio_file": asr_seg["audio_file"],
                    "duration": asr_seg["duration"],
                    "start_time": asr_seg["start_time"],
                    "end_time": asr_seg["end_time"],
                    "speaker": asr_seg["speaker"],
                    "text": asr_seg["text"][:100] + "..."
                    if len(asr_seg["text"]) > 100
                    else asr_seg["text"],
                },
                "fa_segment": {
                    "audio_file": fa_seg["audio_file"],
                    "duration": fa_seg["duration"],
                    "start_time": fa_seg["start_time"],
                    "end_time": fa_seg["end_time"],
                    "speaker": fa_seg["speaker"],
                    "text": fa_seg["text"][:100] + "..."
                    if len(fa_seg["text"]) > 100
                    else fa_seg["text"],
                },
                "significant_difference": duration_diff > 0.1
                or start_diff > 0.1
                or end_diff > 0.1,
            }
            comparison_segments.append(comparison_data)

        elif asr_seg and not fa_seg:
            comparison_segments.append(
                {
                    "segment_index": i + 1,
                    "only_in_asr": True,
                    "asr_segment": asr_seg,
                    "fa_segment": None,
                }
            )

        elif fa_seg and not asr_seg:
            comparison_segments.append(
                {
                    "segment_index": i + 1,
                    "only_in_fa": True,
                    "asr_segment": None,
                    "fa_segment": fa_seg,
                }
            )

    # Generate summary statistics
    total_segments = len(comparison_segments)
    significant_differences = sum(
        1 for seg in comparison_segments if seg.get("significant_difference", False)
    )

    if comparison_segments and all(
        seg.get("duration_difference") is not None
        for seg in comparison_segments
        if "duration_difference" in seg
    ):
        avg_duration_diff = sum(
            seg["duration_difference"]
            for seg in comparison_segments
            if "duration_difference" in seg
        ) / len([seg for seg in comparison_segments if "duration_difference" in seg])
        max_duration_diff = max(
            seg["duration_difference"]
            for seg in comparison_segments
            if "duration_difference" in seg
        )
    else:
        avg_duration_diff = 0
        max_duration_diff = 0

    audio_report = {
        "summary": {
            "total_segments_compared": total_segments,
            "segments_with_significant_differences": significant_differences,
            "significant_difference_rate": (
                significant_differences / total_segments * 100
            )
            if total_segments > 0
            else 0,
            "average_duration_difference": avg_duration_diff,
            "max_duration_difference": max_duration_diff,
        },
        "segments": comparison_segments,
        "listening_guide": {
            "asr_audio_directory": f"{output_dir}/audio_segments/asr_timestamps",
            "fa_audio_directory": f"{output_dir}/audio_segments/forced_alignment_timestamps",
            "recommended_comparison_segments": [
                seg["segment_index"]
                for seg in comparison_segments
                if seg.get("significant_difference", False)
            ][
                :10
            ],  # Top 10 most different segments
        },
    }

    report_file = f"{output_dir}/{base_name}_audio_comparison_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(audio_report, f, indent=2, ensure_ascii=False)

    print(f"\n=== AUDIO COMPARISON SUMMARY ===")
    print(f"Total audio segments compared: {total_segments}")
    print(
        f"Segments with significant differences (>0.1s): {significant_differences} ({significant_differences/total_segments*100:.1f}%)"
    )
    print(f"Average duration difference: {avg_duration_diff:.3f}s")
    print(f"Maximum duration difference: {max_duration_diff:.3f}s")

    if audio_report["listening_guide"]["recommended_comparison_segments"]:
        print(
            f"Recommended segments to listen to (most different): {audio_report['listening_guide']['recommended_comparison_segments']}"
        )

    print(f"Audio comparison report saved to: {report_file}")
    print(
        f"ASR audio segments: {audio_report['listening_guide']['asr_audio_directory']}"
    )
    print(f"FA audio segments: {audio_report['listening_guide']['fa_audio_directory']}")

    return audio_report


def compare_timestamp_methods(audio_path):
    """Compare ASR vs Forced Alignment timestamp methods"""

    print(f"\n=== TIMESTAMP METHOD COMPARISON ===")
    print(f"Audio file: {audio_path}")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = f"timestamp_comparison/{base_name}"
    ensure_dir_exists(f"{output_dir}/dummy.txt")

    # Step 1: Get ASR result
    print("\n=== STEP 1: ASR WITH DIARIZATION ===")
    result = speech_to_text_with_diarization(audio_path)

    if not result or not hasattr(result, "words") or not result.words:
        print("No transcription result or words found")
        return None

    # Step 2: Prepare ASR words
    asr_words = []
    speakers = set()

    for word in result.words:
        word_dict = {
            "text": word.text,
            "type": word.type,
            "start": word.start,
            "end": word.end,
        }
        if hasattr(word, "speaker_id"):
            word_dict["speaker_id"] = word.speaker_id
            if word.type == "word":
                speakers.add(word.speaker_id)
        asr_words.append(word_dict)

    asr_words.sort(key=lambda x: x.get("start", 0))

    # Step 3: Extract turns using ASR timestamps
    print("\n=== STEP 2: EXTRACT TURNS WITH ASR TIMESTAMPS ===")
    asr_turns = extract_turns_from_words(asr_words, "asr_timestamps")

    # Step 4: Get forced alignment
    print("\n=== STEP 3: FORCED ALIGNMENT ===")
    alignment_result = forced_alignment(audio_path, result.text)

    if not alignment_result:
        print("Forced alignment failed - cannot compare methods")
        return None

    alignment_data = {
        "words": [
            {"text": word.text, "start": word.start, "end": word.end}
            for word in alignment_result.words
        ]
        if hasattr(alignment_result, "words")
        else []
    }

    # Step 5: Match words with forced alignment timestamps
    print("\n=== STEP 4: MATCH WITH FORCED ALIGNMENT ===")
    fa_words = match_words_with_forced_alignment(
        asr_words, alignment_data.get("words", [])
    )

    # Step 6: Extract turns using forced alignment timestamps
    print("\n=== STEP 5: EXTRACT TURNS WITH FORCED ALIGNMENT TIMESTAMPS ===")
    fa_turns = extract_turns_from_words(fa_words, "forced_alignment_timestamps")

    # Step 7: Generate audio segments for both methods
    print("\n=== STEP 6: GENERATE AUDIO SEGMENTS ===")
    asr_segments = segment_audio_by_turns(
        audio_path, asr_turns, output_dir, "asr_timestamps"
    )
    fa_segments = segment_audio_by_turns(
        audio_path, fa_turns, output_dir, "forced_alignment_timestamps"
    )

    # Step 8: Compare results
    print("\n=== STEP 7: COMPARE RESULTS ===")
    differences = analyze_turn_differences(asr_turns, fa_turns)

    # Step 9: Create audio comparison report
    print("\n=== STEP 8: AUDIO COMPARISON ANALYSIS ===")
    audio_comparison = create_comparison_audio_report(
        asr_segments, fa_segments, output_dir, base_name
    )

    # Step 10: Save results
    results = {
        "metadata": {
            "audio_file": audio_path,
            "base_name": base_name,
            "language": result.language_code,
            "language_confidence": result.language_probability,
            "total_speakers": len(speakers),
            "speakers": sorted(list(speakers)),
        },
        "asr_method": {
            "total_turns": len(asr_turns),
            "turns": asr_turns,
            "audio_segments": asr_segments,
            "method": "asr_timestamps",
        },
        "forced_alignment_method": {
            "total_turns": len(fa_turns),
            "turns": fa_turns,
            "audio_segments": fa_segments,
            "method": "forced_alignment_timestamps",
        },
        "comparison": {
            "turn_count_difference": len(fa_turns) - len(asr_turns),
            "differences_analyzed": len(differences),
            "audio_comparison": audio_comparison,
        },
    }

    results_file = f"{output_dir}/{base_name}_timestamp_comparison.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {results_file}")

    # Generate detailed comparison report
    if differences:
        comparison_report = generate_comparison_report(
            differences, output_dir, base_name
        )
        results["detailed_comparison"] = comparison_report

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare ASR vs Forced Alignment timestamp methods for turn segmentation"
    )
    parser.add_argument("audio_path", help="Path to the audio file")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}")
        exit(1)

    compare_timestamp_methods(args.audio_path)
