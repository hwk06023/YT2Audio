import os
import argparse
from full_elevenlabs_efficient import process_full_audio_analysis_efficient
from speaker_assignment_efficient import run_efficient_speaker_assignment


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

    print(f"\n{'='*60}")
    print("COMPLETE ANALYSIS FINISHED!")
    print(f"{'='*60}")
    print(f"ElevenLabs result: {elevenlabs_output}")
    print(f"Final result: {final_output}")

    return {
        "success": True,
        "elevenlabs_output": elevenlabs_output,
        "final_output": final_output,
        "elevenlabs_data": unified_data,
        "assignment_data": assignment_result["full_result"],
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
        print(f"\n💡 To generate diarized text:")
        print(f"   python json_to_diarized_text.py \"{result['final_output']}\"")
    else:
        print(f"\n❌ FAILED: {result['error']}")
        exit(1)


if __name__ == "__main__":
    main()
