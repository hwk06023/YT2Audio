# directory_names: List[str], directory_name: str
# Full audio path: "data/{directory_name}.wav"

import os
import json
import io
import re
import subprocess
from typing import List, Dict, Tuple, Optional
from pydub import AudioSegment
from elevenlabs import ElevenLabs
from dotenv import load_dotenv
from textgrids import TextGrid

load_dotenv()


class AudioProcessingPipeline:
    def __init__(self, directory_name: str):
        self.directory_name = directory_name
        self.audio_file = f"data/{directory_name}.wav"
        self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"), timeout=1200)
        self.chunk_duration = 5 * 60 * 1000
        self.output_dir = f"results/{directory_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"temp_chunks/{directory_name}", exist_ok=True)
        os.makedirs(f"mfa_temp/{directory_name}", exist_ok=True)

    def extract_audio_chunk(self, start_ms: int, end_ms: int, chunk_id: int) -> str:
        audio = AudioSegment.from_wav(self.audio_file)
        chunk = audio[start_ms:end_ms]
        chunk_path = f"temp_chunks/{self.directory_name}/chunk_{chunk_id}.wav"
        chunk.export(chunk_path, format="wav")
        return chunk_path

    def transcribe_audio(self, audio_path: str) -> Dict:
        with open(audio_path, "rb") as f:
            buffer = io.BytesIO(f.read())

        result = self.client.speech_to_text.convert(
            file=buffer, model_id="scribe_v1", diarize=True, tag_audio_events=False
        )

        plain_text = ""
        turns = []
        current_speaker = None
        current_text = ""

        if hasattr(result, "words") and result.words:
            word_texts = []
            for word in result.words:
                if word.type == "word":
                    if word.text:
                        word_texts.append(word.text)
                        speaker = (
                            word.speaker_id
                            if hasattr(word, "speaker_id")
                            else "Unknown"
                        )

                        if speaker != current_speaker:
                            if current_text and current_speaker:
                                turns.append(
                                    {
                                        "text": current_text.strip(),
                                        "speaker": current_speaker,
                                    }
                                )
                            current_speaker = speaker
                            current_text = word.text
                        else:
                            current_text += f" {word.text}"

            if current_text and current_speaker:
                turns.append({"text": current_text.strip(), "speaker": current_speaker})
            plain_text = " ".join(word_texts)
        else:
            plain_text = result.text
            turns = [{"text": plain_text, "speaker": "speaker_1"}]

        return {"plain_text": plain_text, "turns": turns}

    # TODO: fix this function
    def find_matching_turn(self, reference_turn: Dict, full_turns: List[Dict]) -> int:
        reference_text = reference_turn["text"].strip()
        for i, turn in enumerate(full_turns):
            if (
                turn["text"].strip() == reference_text
                and turn["speaker"] == reference_turn["speaker"]
            ):
                return i

        for i, turn in enumerate(full_turns):
            if reference_text in turn["text"] or turn["text"] in reference_text:
                if turn["speaker"] == reference_turn["speaker"]:
                    return i

        return -1

    def replace_chunk_transcription(
        self, chunk_turns: List[Dict], full_turns: List[Dict]
    ) -> Tuple[List[Dict], int]:
        if len(chunk_turns) < 2:
            return chunk_turns, -1

        reference_turn = chunk_turns[-2]
        match_index = self.find_matching_turn(reference_turn, full_turns)

        if match_index == -1:
            return chunk_turns, -1

        replaced_turns = full_turns[: match_index + 1]
        if len(chunk_turns) > match_index + 1:
            replaced_turns.extend(chunk_turns[match_index + 1 :])

        return replaced_turns, match_index

    def create_mfa_input(
        self, text: str, audio_path: str, chunk_id: int
    ) -> Tuple[str, str]:
        mfa_dir = f"mfa_temp/{self.directory_name}/chunk_{chunk_id}"
        os.makedirs(mfa_dir, exist_ok=True)

        clean_text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)

        lab_path = f"{mfa_dir}/audio.lab"
        wav_path = f"{mfa_dir}/audio.wav"

        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                wav_path,
            ],
            capture_output=True,
        )

        return mfa_dir, wav_path

    def run_mfa_alignment(self, mfa_dir: str, chunk_id: int) -> Optional[str]:
        align_dir = f"mfa_temp/{self.directory_name}/chunk_{chunk_id}_align"
        os.makedirs(align_dir, exist_ok=True)

        try:
            subprocess.run(
                [
                    "mfa",
                    "validate",
                    mfa_dir,
                    "korean_mfa",
                    "--clean",
                    "--beam",
                    "40000",
                ],
                capture_output=True,
                check=True,
            )

            subprocess.run(
                [
                    "mfa",
                    "align",
                    mfa_dir,
                    "korean_mfa",
                    "korean_mfa",
                    align_dir,
                    "--beam",
                    "40000",
                ],
                capture_output=True,
                check=True,
            )

            textgrid_path = f"{align_dir}/audio.TextGrid"
            return textgrid_path if os.path.exists(textgrid_path) else None
        except subprocess.CalledProcessError:
            return None

    def parse_textgrid_timestamps(
        self, textgrid_path: str
    ) -> Dict[str, Tuple[float, float]]:
        try:
            tg = TextGrid(textgrid_path)
            word_timestamps = {}

            for tier in tg:
                if tier.name == "words":
                    for interval in tier:
                        if interval.text.strip():
                            word_timestamps[interval.text.strip()] = (
                                interval.xmin,
                                interval.xmax,
                            )

            return word_timestamps
        except:
            return {}

    def apply_timestamps_to_turns(
        self,
        turns: List[Dict],
        word_timestamps: Dict[str, Tuple[float, float]],
        time_offset: float = 0,
    ) -> List[Dict]:
        timestamped_turns = []
        current_time = time_offset

        for turn in turns:
            words = turn["text"].split()
            turn_start = current_time
            turn_end = current_time

            for word in words:
                clean_word = re.sub(r"[^가-힣a-zA-Z0-9]", "", word)
                if clean_word in word_timestamps:
                    word_start, word_end = word_timestamps[clean_word]
                    turn_end = max(turn_end, word_start + word_end + time_offset)
                else:
                    turn_end += 0.5

            current_time = turn_end

            timestamped_turns.append(
                {
                    "text": turn["text"],
                    "speaker": turn["speaker"],
                    "start_time": f"{turn_start:.2f}",
                    "end_time": f"{turn_end:.2f}",
                }
            )

        return timestamped_turns

    def process_full_audio(self) -> List[Dict]:
        print(f"Processing full audio transcription for {self.directory_name}")
        full_transcription = self.transcribe_audio(self.audio_file)

        output_path = f"{self.output_dir}/full_transcription.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_transcription, f, ensure_ascii=False, indent=2)

        return full_transcription["turns"]

    def process_pipeline(self) -> str:
        print("Starting integrated audio processing pipeline...")

        full_turns = self.process_full_audio()
        audio = AudioSegment.from_wav(self.audio_file)
        total_duration = len(audio)

        all_timestamped_turns = []
        current_start = 0
        chunk_id = 0

        while current_start < total_duration:
            current_end = min(current_start + self.chunk_duration, total_duration)

            print(
                f"Processing chunk {chunk_id}: {current_start/1000:.1f}s - {current_end/1000:.1f}s"
            )

            chunk_path = self.extract_audio_chunk(current_start, current_end, chunk_id)
            chunk_transcription = self.transcribe_audio(chunk_path)
            chunk_turns = chunk_transcription["turns"]

            replaced_turns, match_index = self.replace_chunk_transcription(
                chunk_turns, full_turns
            )

            if match_index == -1:
                print(f"Warning: Could not find matching turn for chunk {chunk_id}")
                replaced_turns = chunk_turns

            combined_text = " ".join([turn["text"] for turn in replaced_turns])
            mfa_dir, mfa_wav = self.create_mfa_input(
                combined_text, chunk_path, chunk_id
            )

            textgrid_path = self.run_mfa_alignment(mfa_dir, chunk_id)

            if textgrid_path:
                word_timestamps = self.parse_textgrid_timestamps(textgrid_path)
                timestamped_turns = self.apply_timestamps_to_turns(
                    replaced_turns, word_timestamps, current_start / 1000
                )
            else:
                print(
                    f"MFA alignment failed for chunk {chunk_id}, using estimated timestamps"
                )
                timestamped_turns = self.apply_timestamps_to_turns(
                    replaced_turns, {}, current_start / 1000
                )

            all_timestamped_turns.extend(timestamped_turns)

            if match_index != -1 and len(timestamped_turns) > 0:
                last_turn_end = float(timestamped_turns[-1]["end_time"])
                current_start = int(last_turn_end * 1000)
            else:
                current_start = current_end

            chunk_id += 1

        final_result = {"turns": all_timestamped_turns}
        output_path = f"{self.output_dir}/final_timestamped_transcription.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        print(f"Pipeline completed! Results saved to {output_path}")
        return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Audio Processing Pipeline with Speaker Diarization and Timestamp Alignment"
    )
    parser.add_argument(
        "directory_name", help="Directory name containing the audio file"
    )
    args = parser.parse_args()

    pipeline = AudioProcessingPipeline(args.directory_name)
    result_path = pipeline.process_pipeline()
    print(f"Final results available at: {result_path}")


if __name__ == "__main__":
    main()
