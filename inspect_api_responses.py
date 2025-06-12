import os
import json
import argparse
from dotenv import load_dotenv
import io
import time
from pydub import AudioSegment
from elevenlabs import ElevenLabs
import httpx
import difflib
import re

load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"), timeout=300.0)


def normalize_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower().strip())
    return text


def calculate_text_similarity(text1, text2):
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    if norm1 == norm2:
        return 1.0

    if not norm1 or not norm2:
        return 0.0

    similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
    return similarity


def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    return file_path


def speech_to_text_raw(audio_path, max_retries=3):
    print(f"=== ASR API CALL ===")
    print(f"Processing audio file: {audio_path}")

    audio = AudioSegment.from_file(audio_path)
    duration_minutes = len(audio) / (1000 * 60)
    print(f"Audio duration: {duration_minutes:.2f} minutes")

    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)

    for attempt in range(max_retries):
        try:
            print(f"Calling ElevenLabs ASR (attempt {attempt + 1}/{max_retries})...")
            result = client.speech_to_text.convert(
                file=buffer,
                model_id="scribe_v1_experimental",
                diarize=True,
                tag_audio_events=True,
            )
            print("ASR call successful!")
            return result

        except httpx.ReadTimeout as e:
            print(f"ASR timeout on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                buffer.seek(0)
            else:
                raise e
        except Exception as e:
            print(f"ASR error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                buffer.seek(0)
            else:
                raise e


def forced_alignment_raw(audio_path, transcript_text, max_retries=3):
    print(f"\n=== FORCED ALIGNMENT API CALL ===")
    print(f"Processing audio file: {audio_path}")
    print(f"Transcript length: {len(transcript_text)} characters")

    audio = AudioSegment.from_file(audio_path)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)

    for attempt in range(max_retries):
        try:
            print(
                f"Calling ElevenLabs Forced Alignment (attempt {attempt + 1}/{max_retries})..."
            )
            alignment_result = client.forced_alignment.create(
                file=buffer, text=transcript_text
            )
            print("Forced Alignment call successful!")
            return alignment_result

        except httpx.ReadTimeout as e:
            print(f"Forced Alignment timeout on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                buffer.seek(0)
            else:
                raise e
        except Exception as e:
            print(f"Forced Alignment error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                buffer.seek(0)
            else:
                raise e


def extract_diarized_turns_from_asr(asr_result):
    if not hasattr(asr_result, "words") or not asr_result.words:
        print("No word-level data available for turn extraction")
        return []

    turns = []
    current_speaker = None
    current_turn_words = []

    for word in asr_result.words:
        if (
            hasattr(word, "type")
            and word.type == "word"
            and hasattr(word, "text")
            and word.text
        ):
            speaker = getattr(word, "speaker_id", "Unknown")

            if speaker != current_speaker:
                if current_turn_words:
                    turn_text = " ".join([w.text for w in current_turn_words])
                    turns.append(
                        {
                            "speaker": current_speaker,
                            "asr_start_time": current_turn_words[0].start,
                            "asr_end_time": current_turn_words[-1].end,
                            "text": turn_text,
                            "word_count": len(current_turn_words),
                        }
                    )
                current_speaker = speaker
                current_turn_words = [word]
            else:
                current_turn_words.append(word)

    if current_turn_words:
        turn_text = " ".join([w.text for w in current_turn_words])
        turns.append(
            {
                "speaker": current_speaker,
                "asr_start_time": current_turn_words[0].start,
                "asr_end_time": current_turn_words[-1].end,
                "text": turn_text,
                "word_count": len(current_turn_words),
            }
        )

    print(f"Extracted {len(turns)} diarized turns from ASR")
    for i, turn in enumerate(turns):
        print(
            f"  Turn {i+1}: {turn['speaker']} ({turn['asr_start_time']:.2f}s - {turn['asr_end_time']:.2f}s) - {turn['word_count']} words"
        )

    return turns


def clean_text_for_matching(text):
    """매칭을 위해 텍스트를 정규화: 공백과 대괄호 안의 내용 제거"""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def match_turns_with_json_files(turn_json_file, asr_json_file, fa_json_file):
    """턴 JSON을 기준으로 ASR과 FA JSON 파일과 매칭"""
    print(f"\n=== MATCHING TURNS WITH JSON FILES ===")

    with open(turn_json_file, "r", encoding="utf-8") as f:
        turn_data = json.load(f)

    with open(asr_json_file, "r", encoding="utf-8") as f:
        asr_data = json.load(f)

    with open(fa_json_file, "r", encoding="utf-8") as f:
        fa_data = json.load(f)

    turns = turn_data.get("detailed_turns", [])
    asr_words = [w for w in asr_data.get("words", []) if w.get("type") == "word"]
    fa_words = fa_data.get("words", [])

    print(f"Turns: {len(turns)}")
    print(f"ASR words: {len(asr_words)}")
    print(f"FA words: {len(fa_words)}")

    enhanced_turns = []
    asr_idx = 0
    fa_idx = 0

    for turn_idx, turn in enumerate(turns):
        print(f"\nProcessing turn {turn_idx + 1}: {turn.get('speaker', 'Unknown')}")

        turn_text = turn.get("text", "")
        turn_clean = clean_text_for_matching(turn_text)
        print(f"  Turn text (cleaned): {turn_clean[:50]}...")

        # ASR 매칭
        asr_start_idx = asr_idx
        asr_matched_text = ""
        asr_start_time = None
        asr_end_time = None

        while asr_idx < len(asr_words):
            word = asr_words[asr_idx]
            word_clean = clean_text_for_matching(word.get("text", ""))

            if asr_start_time is None:
                asr_start_time = word.get("start")

            asr_matched_text += word_clean
            asr_end_time = word.get("end")

            if turn_clean.startswith(asr_matched_text):
                if turn_clean == asr_matched_text:
                    asr_idx += 1
                    break
                asr_idx += 1
            else:
                break

        print(f"  ASR matched words: {asr_idx - asr_start_idx}")
        if asr_start_time and asr_end_time:
            print(f"  ASR timing: {asr_start_time:.3f}s - {asr_end_time:.3f}s")

        # FA 매칭
        fa_start_idx = fa_idx
        fa_matched_text = ""
        fa_start_time = None
        fa_end_time = None

        while fa_idx < len(fa_words):
            word = fa_words[fa_idx]
            word_clean = clean_text_for_matching(word.get("text", ""))

            if fa_start_time is None:
                fa_start_time = word.get("start")

            fa_matched_text += word_clean
            fa_end_time = word.get("end")

            if turn_clean.startswith(fa_matched_text):
                if turn_clean == fa_matched_text:
                    fa_idx += 1
                    break
                fa_idx += 1
            else:
                break

        print(f"  FA matched words: {fa_idx - fa_start_idx}")
        if fa_start_time and fa_end_time:
            print(f"  FA timing: {fa_start_time:.3f}s - {fa_end_time:.3f}s")

        enhanced_turn = {
            "speaker": turn.get("speaker", "Unknown"),
            "text": turn_text,
            "asr_start_time": asr_start_time,
            "asr_end_time": asr_end_time,
            "fa_start_time": fa_start_time,
            "fa_end_time": fa_end_time,
            "asr_matched_words": asr_idx - asr_start_idx,
            "fa_matched_words": fa_idx - fa_start_idx,
        }

        enhanced_turns.append(enhanced_turn)

    print(f"\nMatching complete: {len(enhanced_turns)} turns processed")
    print(f"ASR words used: {asr_idx}/{len(asr_words)}")
    print(f"FA words used: {fa_idx}/{len(fa_words)}")

    return enhanced_turns


def segment_audio_by_timing(audio_path, enhanced_turns, output_dir):
    print(f"\n=== SEGMENTING AUDIO BY TIMING ===")

    audio = AudioSegment.from_file(audio_path)

    asr_dir = f"{output_dir}/asr"
    fa_dir = f"{output_dir}/fa"
    ensure_dir_exists(f"{asr_dir}/dummy.wav")
    ensure_dir_exists(f"{fa_dir}/dummy.wav")

    for i, turn in enumerate(enhanced_turns):
        turn_num = f"{i+1:03d}"

        if turn.get("asr_start_time") and turn.get("asr_end_time"):
            asr_start_ms = int(turn["asr_start_time"] * 1000)
            asr_end_ms = int(turn["asr_end_time"] * 1000)
            asr_audio = audio[asr_start_ms:asr_end_ms]
            asr_file = f"{asr_dir}/{turn_num}_{turn['speaker']}.wav"
            asr_audio.export(asr_file, format="wav")
            print(
                f"ASR Turn {turn_num}: {turn['speaker']} ({turn['asr_start_time']:.2f}s - {turn['asr_end_time']:.2f}s) -> {asr_file}"
            )

        if turn.get("fa_start_time") and turn.get("fa_end_time"):
            fa_start_ms = int(turn["fa_start_time"] * 1000)
            fa_end_ms = int(turn["fa_end_time"] * 1000)
            fa_audio = audio[fa_start_ms:fa_end_ms]
            fa_file = f"{fa_dir}/{turn_num}_{turn['speaker']}.wav"
            fa_audio.export(fa_file, format="wav")
            print(
                f"FA Turn {turn_num}: {turn['speaker']} ({turn['fa_start_time']:.2f}s - {turn['fa_end_time']:.2f}s) -> {fa_file}"
            )

    print(f"Audio segmentation complete!")
    print(f"ASR segments saved to: {asr_dir}")
    print(f"FA segments saved to: {fa_dir}")


def serialize_asr_result(asr_result):
    """ASR 결과를 JSON 직렬화 가능한 딕셔너리로 변환"""
    if not asr_result:
        return None

    result_dict = {"transcript": getattr(asr_result, "transcript", ""), "words": []}

    if hasattr(asr_result, "words") and asr_result.words:
        for word in asr_result.words:
            word_dict = {
                "text": getattr(word, "text", ""),
                "start": getattr(word, "start", 0),
                "end": getattr(word, "end", 0),
                "type": getattr(word, "type", ""),
                "speaker_id": getattr(word, "speaker_id", None),
            }
            result_dict["words"].append(word_dict)

    return result_dict


def serialize_fa_result(fa_result):
    """FA 결과를 JSON 직렬화 가능한 딕셔너리로 변환"""
    if not fa_result:
        return None

    result_dict = {"words": []}

    if hasattr(fa_result, "words") and fa_result.words:
        for word in fa_result.words:
            word_dict = {
                "text": getattr(word, "text", ""),
                "start": getattr(word, "start", 0),
                "end": getattr(word, "end", 0),
            }
            result_dict["words"].append(word_dict)

    return result_dict


def inspect_api_responses(audio_path):
    print(f"=== API RESPONSE INSPECTOR ===")
    print(f"Audio file: {audio_path}")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = f"api_inspection/{base_name}"
    ensure_dir_exists(f"{output_dir}/dummy.txt")

    asr_result = speech_to_text_raw(audio_path)

    asr_json_file = f"{output_dir}/{base_name}_asr_result.json"
    serialized_asr = serialize_asr_result(asr_result)
    with open(asr_json_file, "w", encoding="utf-8") as f:
        json.dump(serialized_asr, f, indent=2, ensure_ascii=False)
    print(f"ASR result saved to: {asr_json_file}")

    print("\n=== EXTRACTING DIARIZED TURNS ===")
    diarized_turns = extract_diarized_turns_from_asr(asr_result)

    if hasattr(asr_result, "words") and asr_result.words:
        word_texts = []
        for word in asr_result.words:
            if (
                hasattr(word, "type")
                and word.type == "word"
                and hasattr(word, "text")
                and word.text
            ):
                word_texts.append(word.text)

        clean_transcript = " ".join(word_texts)
        print(f"Clean transcript length: {len(clean_transcript)} chars (words only)")
        print(f"Total words for alignment: {len(word_texts)}")

        if clean_transcript.strip():
            fa_result = forced_alignment_raw(audio_path, clean_transcript)

            fa_json_file = f"{output_dir}/{base_name}_fa_result.json"
            serialized_fa = serialize_fa_result(fa_result)
            with open(fa_json_file, "w", encoding="utf-8") as f:
                json.dump(serialized_fa, f, indent=2, ensure_ascii=False)
            print(f"FA result saved to: {fa_json_file}")
        else:
            print("No valid words found for forced alignment")
            fa_result = None
    else:
        print("No word-level data available for forced alignment")
        fa_result = None

    if fa_result:
        # 먼저 기본 턴 정보를 저장
        basic_report = {"detailed_turns": diarized_turns}
        report_file = f"{output_dir}/{base_name}_turn_alignment_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(basic_report, f, indent=2, ensure_ascii=False)
        print(f"Basic turn report saved to: {report_file}")

        print("\n=== MATCHING TURNS WITH JSON FILES ===")
        enhanced_turns = match_turns_with_json_files(
            report_file,
            asr_json_file,
            fa_json_file,
        )

        # 향상된 턴 정보로 업데이트
        enhanced_report = {"detailed_turns": enhanced_turns}
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(enhanced_report, f, indent=2, ensure_ascii=False)

        print(f"Enhanced turn alignment report saved to: {report_file}")

        segment_audio_by_timing(audio_path, enhanced_turns, output_dir)

        print(f"\n=== INSPECTION COMPLETE ===")
        print(f"Files created:")
        print(f"1. ASR Result JSON: {asr_json_file}")
        print(f"2. FA Result JSON: {fa_json_file}")
        print(f"3. Turn Alignment Report: {report_file}")
        print(f"4. ASR Audio Segments: {output_dir}/asr/")
        print(f"5. FA Audio Segments: {output_dir}/fa/")

        return {
            "enhanced_turns": enhanced_turns,
            "files": {
                "asr_result_json": asr_json_file,
                "fa_result_json": fa_json_file,
                "turn_alignment_report": report_file,
                "asr_segments_dir": f"{output_dir}/asr",
                "fa_segments_dir": f"{output_dir}/fa",
            },
        }
    else:
        print("No forced alignment data available")
        return {
            "enhanced_turns": [],
            "files": {
                "asr_result_json": asr_json_file,
                "fa_result_json": None,
                "turn_alignment_report": None,
            },
        }


def analyze_timestamp_errors(enhanced_turns):
    print(f"\n=== TIMESTAMP ERROR ANALYSIS ===")

    error_data = []

    for i, turn in enumerate(enhanced_turns):
        if (
            turn.get("fa_start_time") is not None
            and turn.get("fa_end_time") is not None
            and turn.get("asr_start_time") is not None
            and turn.get("asr_end_time") is not None
        ):
            start_error = abs(turn["asr_start_time"] - turn["fa_start_time"])
            end_error = abs(turn["asr_end_time"] - turn["fa_end_time"])
            total_error = start_error + end_error

            error_data.append(
                {
                    "turn_index": i + 1,
                    "speaker": turn["speaker"],
                    "start_error": start_error,
                    "end_error": end_error,
                    "total_error": total_error,
                    "asr_timing": (turn["asr_start_time"], turn["asr_end_time"]),
                    "fa_timing": (turn["fa_start_time"], turn["fa_end_time"]),
                    "text_preview": turn["text"][:50] + "..."
                    if len(turn["text"]) > 50
                    else turn["text"],
                }
            )

    error_data.sort(key=lambda x: x["total_error"], reverse=True)

    print(f"Top 3 turns with largest timestamp errors:")
    print("-" * 80)

    for i, turn_data in enumerate(error_data[:3]):
        print(f"\n{i+1}. Turn {turn_data['turn_index']} ({turn_data['speaker']})")
        print(f"   Text: {turn_data['text_preview']}")
        print(
            f"   ASR timing:  {turn_data['asr_timing'][0]:.3f}s - {turn_data['asr_timing'][1]:.3f}s"
        )
        print(
            f"   FA timing:   {turn_data['fa_timing'][0]:.3f}s - {turn_data['fa_timing'][1]:.3f}s"
        )
        print(f"   Start error: {turn_data['start_error']:.3f}s")
        print(f"   End error:   {turn_data['end_error']:.3f}s")
        print(f"   Total error: {turn_data['total_error']:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect API responses and segment audio by timing"
    )
    parser.add_argument("audio_path", help="Path to the audio file")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}")
        exit(1)

    result = inspect_api_responses(args.audio_path)

    if result["enhanced_turns"]:
        analyze_timestamp_errors(result["enhanced_turns"])
