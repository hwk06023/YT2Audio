import os
import argparse
import json
from pydub import AudioSegment
from full_elevenlabs_efficient import process_full_audio_analysis_efficient
from speaker_assignment_efficient import run_efficient_speaker_assignment


def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    return file_path


def segment_audio_by_final_results(audio_path, final_results_file):
    print(f"\n{'='*40}")
    print("STEP 3: Audio Segmentation by Turns")
    print(f"{'='*40}")

    with open(final_results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    turns = results.get("turns", [])
    if not turns:
        print("No turns found in final results")
        return None

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    model_type = results.get("metadata", {}).get("model_type", "unknown")
    segments_dir = f"audio_segments/{base_name}_{model_type}"
    ensure_dir_exists(f"{segments_dir}/dummy.wav")

    audio = AudioSegment.from_file(audio_path)
    segment_info = []

    print(f"Segmenting audio into {len(turns)} turns...")

    for i, turn in enumerate(turns):
        speaker = turn.get("speaker", "Unknown")
        start_time = turn.get("start_time", 0)
        end_time = turn.get("end_time", 0)
        text = turn.get("text", "")

        if start_time >= end_time:
            print(f"Skipping turn {i+1}: Invalid timing ({start_time}s - {end_time}s)")
            continue

        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        duration = end_time - start_time

        turn_audio = audio[start_ms:end_ms]

        output_file = f"{segments_dir}/{i+1:03d}_{speaker}_{duration:.2f}s.wav"
        turn_audio.export(output_file, format="wav")

        segment_data = {
            "turn_id": i + 1,
            "speaker": speaker,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "text": text,
            "audio_file": output_file,
            "file_size_mb": round(os.path.getsize(output_file) / (1024 * 1024), 2),
        }

        segment_info.append(segment_data)

        print(
            f"Turn {i+1:03d}: {speaker} ({start_time:.2f}s - {end_time:.2f}s) -> {output_file}"
        )

    segments_info_file = f"{segments_dir}/segments_info.json"
    with open(segments_info_file, "w", encoding="utf-8") as f:
        json.dump(segment_info, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Audio segmentation complete!")
    print(f"📁 Segments directory: {segments_dir}")
    print(f"📄 Segments info: {segments_info_file}")
    print(f"🎵 Total segments created: {len(segment_info)}")

    return {
        "segments_dir": segments_dir,
        "segments_info_file": segments_info_file,
        "segments_count": len(segment_info),
        "segments_info": segment_info,
    }


def run_complete_analysis(audio_path, model_type="ecapa"):
    print(f"\n{'='*60}")
    print(f"COMPLETE AUDIO ANALYSIS PIPELINE")
    print(f"{'='*60}")
    print(f"Audio file: {audio_path}")
    print(f"Model: {model_type.upper()}")

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return {"success": False, "error": f"Audio file not found: {audio_path}"}

    print(f"\n{'='*40}")
    print("STEP 1: ElevenLabs Speech-to-Text Analysis")
    print(f"{'='*40}")

    unified_data = process_full_audio_analysis_efficient(audio_path)
    if not unified_data:
        print("Error: Failed to process audio with ElevenLabs")
        return {"success": False, "error": "ElevenLabs processing failed"}

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    elevenlabs_output = f"elevenlabs_results/{base_name}.json"

    print(f"\n{'='*40}")
    print("STEP 2: Speaker Assignment Analysis")
    print(f"{'='*40}")

    assignment_result = run_efficient_speaker_assignment(elevenlabs_output, model_type)
    if not assignment_result["success"]:
        print("Error: Failed to process speaker assignments")
        return {"success": False, "error": "Speaker assignment failed"}

    final_output = assignment_result["output_file"]

    segmentation_result = segment_audio_by_final_results(audio_path, final_output)

    print(f"\n{'='*60}")
    print("COMPLETE ANALYSIS FINISHED!")
    print(f"{'='*60}")
    print(f"ElevenLabs result: {elevenlabs_output}")
    print(f"Final result: {final_output}")
    if segmentation_result:
        print(f"Audio segments: {segmentation_result['segments_dir']}")

    return {
        "success": True,
        "elevenlabs_output": elevenlabs_output,
        "final_output": final_output,
        "elevenlabs_data": unified_data,
        "assignment_data": assignment_result["full_result"],
        "segmentation_result": segmentation_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Complete audio analysis pipeline: ElevenLabs + Speaker Assignment"
    )
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument(
        "--model",
        choices=["ecapa", "wavlm"],
        default="ecapa",
        help="Model type for speaker assignment: ecapa (ECAPA-TDNN) or wavlm (WavLM-based)",
    )

    args = parser.parse_args()

    result = run_complete_analysis(args.audio_path, args.model)

    if result["success"]:
        print(f"\n✅ SUCCESS!")
        print(f"📁 ElevenLabs output: {result['elevenlabs_output']}")
        print(f"📁 Final output: {result['final_output']}")

        if result.get("segmentation_result"):
            seg_result = result["segmentation_result"]
            print(f"🎵 Audio segments: {seg_result['segments_dir']}")
            print(f"📊 Segments created: {seg_result['segments_count']}")
            print(f"📄 Segments info: {seg_result['segments_info_file']}")

        print(f"\n💡 To generate diarized text:")
        print(f"   python json_to_diarized_text.py \"{result['final_output']}\"")
    else:
        print(f"\n❌ FAILED: {result['error']}")
        exit(1)


if __name__ == "__main__":
    main()
