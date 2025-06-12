import json
import argparse
import os
from pathlib import Path


def json_to_diarized_text(json_file_path, output_file=None):
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found: {json_file_path}")
        return None

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "turns" not in data:
        print("Error: 'turns' key not found in JSON data")
        return None

    turns = data["turns"]
    diarized_lines = []

    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        diarized_lines.append(f"{speaker}: {text}")

    diarized_text = "\n".join(diarized_lines)

    if output_file is None:
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_file = f"{base_name}_diarized.txt"

    output_path = os.path.dirname(json_file_path)
    if output_path:
        output_file = os.path.join(output_path, os.path.basename(output_file))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(diarized_text)

    print(f"Diarized text saved to: {output_file}")
    print(f"Total turns: {len(turns)}")

    speaker_counts = {}
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

    print("Speaker distribution:")
    for speaker, count in sorted(speaker_counts.items()):
        print(f"  {speaker}: {count} turns")

    return output_file


def process_directory(directory_path):
    results_dir = Path(directory_path)
    if not results_dir.exists():
        print(f"Error: Directory not found: {directory_path}")
        return

    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in: {directory_path}")
        return

    print(f"Found {len(json_files)} JSON files in {directory_path}")

    for json_file in json_files:
        print(f"\nProcessing: {json_file}")
        json_to_diarized_text(str(json_file))


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON results to diarized text format"
    )
    parser.add_argument(
        "input_path", help="Path to JSON file or directory containing JSON files"
    )
    parser.add_argument(
        "--output", "-o", help="Output text file path (for single file only)"
    )

    args = parser.parse_args()

    if os.path.isfile(args.input_path):
        json_to_diarized_text(args.input_path, args.output)
    elif os.path.isdir(args.input_path):
        if args.output:
            print("Warning: --output ignored when processing directory")
        process_directory(args.input_path)
    else:
        print(f"Error: Path not found: {args.input_path}")


if __name__ == "__main__":
    main()
