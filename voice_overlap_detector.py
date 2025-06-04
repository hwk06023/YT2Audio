import numpy as np
import librosa
import os
import soundfile as sf
from typing import List, Tuple, Optional
import torch
from nemo.collections.asr.models import ClusteringDiarizer
import tempfile
import json
from omegaconf import OmegaConf


class NeMoVoiceOverlapDetector:
    def __init__(self):
        pass

    def create_manifest(self, audio_file: str, temp_dir: str) -> str:
        manifest_path = os.path.join(temp_dir, "input_manifest.json")

        with open(manifest_path, "w") as f:
            audio_entry = {
                "audio_filepath": os.path.abspath(audio_file),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "rttm_filepath": None,
                "uem_filepath": None,
            }
            f.write(json.dumps(audio_entry) + "\n")

        return manifest_path

    def detect_overlaps_with_nemo(self, audio_file: str) -> dict:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_data, sr = librosa.load(audio_file, sr=16000, mono=True)

            temp_audio_file = os.path.join(temp_dir, "temp_mono_audio.wav")
            sf.write(temp_audio_file, audio_data, sr)

            manifest_path = self.create_manifest(temp_audio_file, temp_dir)

            device = "cpu"

            config = OmegaConf.create(
                {
                    "device": device,
                    "sample_rate": 16000,
                    "num_workers": 0,
                    "verbose": True,
                    "batch_size": 1,
                    "diarizer": {
                        "manifest_filepath": manifest_path,
                        "out_dir": temp_dir,
                        "oracle_vad": False,
                        "collar": 0.25,
                        "ignore_overlap": False,
                        "vad": {
                            "model_path": "vad_multilingual_marblenet",
                            "parameters": {
                                "window_length_in_sec": 0.15,
                                "shift_length_in_sec": 0.01,
                                "smoothing": "median",
                                "overlap": 0.875,
                                "onset": 0.3,
                                "offset": 0.6,
                                "pad_onset": 0.1,
                                "pad_offset": 0.1,
                                "min_duration_on": 0.1,
                                "min_duration_off": 0.1,
                                "filter_speech_first": False,
                            },
                        },
                        "speaker_embeddings": {
                            "model_path": "titanet_large",
                            "parameters": {
                                "window_length_in_sec": 1.0,
                                "shift_length_in_sec": 0.5,
                                "multiscale_weights": None,
                                "save_embeddings": False,
                            },
                        },
                        "clustering": {
                            "parameters": {
                                "oracle_num_speakers": False,
                                "max_num_speakers": 20,
                                "enhanced_count_thres": 40,
                                "max_rp_threshold": 0.15,
                                "sparse_search_volume": 50,
                            }
                        },
                    },
                }
            )

            model = ClusteringDiarizer(cfg=config)
            model.diarize()

            rttm_file = os.path.join(
                temp_dir,
                "pred_rttms",
                os.path.basename(temp_audio_file).replace(".wav", ".rttm"),
            )

            if os.path.exists(rttm_file):
                return self.parse_rttm_results(rttm_file, audio_file)
            else:
                return {"error": "RTTM file not generated"}

    def parse_rttm_results(self, rttm_file: str, audio_file: str) -> dict:
        audio_data, sr = librosa.load(audio_file, sr=None)
        total_duration = len(audio_data) / sr

        speaker_segments = {}
        overlap_segments = []

        with open(rttm_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    end_time = start_time + duration
                    speaker_id = parts[7]

                    if speaker_id not in speaker_segments:
                        speaker_segments[speaker_id] = []
                    speaker_segments[speaker_id].append((start_time, end_time))

        speakers = list(speaker_segments.keys())
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                speaker1_segments = speaker_segments[speakers[i]]
                speaker2_segments = speaker_segments[speakers[j]]

                for start1, end1 in speaker1_segments:
                    for start2, end2 in speaker2_segments:
                        overlap_start = max(start1, start2)
                        overlap_end = min(end1, end2)
                        if overlap_start < overlap_end:
                            overlap_segments.append((overlap_start, overlap_end))

        overlap_segments = self.merge_overlapping_segments(overlap_segments)

        return {
            "total_duration": total_duration,
            "speaker_segments": speaker_segments,
            "overlap_segments": overlap_segments,
            "overlap_duration": sum(end - start for start, end in overlap_segments),
            "overlap_ratio": sum(end - start for start, end in overlap_segments)
            / total_duration
            if total_duration > 0
            else 0,
            "num_speakers": len(speakers),
        }

    def merge_overlapping_segments(
        self, segments: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        if not segments:
            return []

        sorted_segments = sorted(segments)
        merged = [sorted_segments[0]]

        for current_start, current_end in sorted_segments[1:]:
            last_start, last_end = merged[-1]

            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged

    def get_non_overlap_segments(
        self, total_duration: float, overlap_segments: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        if not overlap_segments:
            return [(0.0, total_duration)]

        non_overlap_segments = []
        sorted_overlaps = sorted(overlap_segments)

        if sorted_overlaps[0][0] > 0:
            non_overlap_segments.append((0.0, sorted_overlaps[0][0]))

        for i in range(len(sorted_overlaps) - 1):
            gap_start = sorted_overlaps[i][1]
            gap_end = sorted_overlaps[i + 1][0]
            if gap_start < gap_end:
                non_overlap_segments.append((gap_start, gap_end))

        if sorted_overlaps[-1][1] < total_duration:
            non_overlap_segments.append((sorted_overlaps[-1][1], total_duration))

        return non_overlap_segments

    def save_audio_segments(
        self,
        audio_file: str,
        audio_name: str,
        overlap_segments: List[Tuple[float, float]],
        total_duration: float,
    ):
        audio_data, sr = librosa.load(audio_file, sr=None, mono=False)

        if len(audio_data.shape) == 1:
            audio_data = np.array([audio_data])

        overlap_dir = f"is_overlap/{audio_name}/true"
        non_overlap_dir = f"is_overlap/{audio_name}/false"

        os.makedirs(overlap_dir, exist_ok=True)
        os.makedirs(non_overlap_dir, exist_ok=True)

        for i, (start, end) in enumerate(overlap_segments):
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            if len(audio_data.shape) == 2:
                segment = audio_data[:, start_sample:end_sample]
            else:
                segment = audio_data[start_sample:end_sample]

            output_file = f"{overlap_dir}/overlap_{i+1:03d}_{start:.2f}s_{end:.2f}s.wav"
            sf.write(output_file, segment.T if len(segment.shape) == 2 else segment, sr)

        non_overlap_segments = self.get_non_overlap_segments(
            total_duration, overlap_segments
        )

        for i, (start, end) in enumerate(non_overlap_segments):
            if end - start < 0.1:
                continue

            start_sample = int(start * sr)
            end_sample = int(end * sr)

            if len(audio_data.shape) == 2:
                segment = audio_data[:, start_sample:end_sample]
            else:
                segment = audio_data[start_sample:end_sample]

            output_file = (
                f"{non_overlap_dir}/non_overlap_{i+1:03d}_{start:.2f}s_{end:.2f}s.wav"
            )
            sf.write(output_file, segment.T if len(segment.shape) == 2 else segment, sr)


def main():
    detector = NeMoVoiceOverlapDetector()

    audio_name = "gapyear"
    audio_file = "data/" + audio_name + ".wav"

    try:
        print("Loading NeMo ClusteringDiarizer and processing audio...")
        results = detector.detect_overlaps_with_nemo(audio_file)

        if "error" in results:
            print(f"Error: {results['error']}")
            return

        print(f"Total Duration: {results['total_duration']:.2f} seconds")
        print(f"Number of Speakers: {results['num_speakers']}")
        print(f"Overlap Duration: {results['overlap_duration']:.2f} seconds")
        print(f"Overlap Ratio: {results['overlap_ratio']:.2%}")

        print(f"\nSpeaker Segments:")
        for speaker_id, segments in results["speaker_segments"].items():
            print(f"  Speaker {speaker_id}: {len(segments)} segments")
            for i, (start, end) in enumerate(segments[:3]):
                print(f"    {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
            if len(segments) > 3:
                print(f"    ... and {len(segments)-3} more segments")

        print(f"\nOverlap Segments:")
        for i, (start, end) in enumerate(results["overlap_segments"]):
            print(f"  {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")

        detector.save_audio_segments(
            audio_file,
            audio_name,
            results["overlap_segments"],
            results["total_duration"],
        )

        print(f"\nAudio segments saved to:")
        print(f"  Overlap segments: is_overlap/{audio_name}/true/")
        print(f"  Non-overlap segments: is_overlap/{audio_name}/false/")

    except FileNotFoundError:
        print("Audio file not found. Please provide a valid audio file.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
