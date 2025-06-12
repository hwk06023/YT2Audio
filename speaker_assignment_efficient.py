import os
import json
import numpy as np
import torch
import torchaudio
import librosa
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import argparse
from collections import defaultdict
from pathlib import Path
from speechbrain.inference import EncoderClassifier
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from pydub import AudioSegment
import tempfile


class SpeakerEmbeddingAssignerEfficient:
    def __init__(self, model_type="ecapa", device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_type = model_type
        print(f"Using device: {self.device}")
        print(f"Using model type: {model_type}")

        self._init_ecapa_model()

    def _init_ecapa_model(self):
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )
        self.target_sr = 16000
        print("ECAPA-TDNN model loaded successfully")

    def extract_audio_segment(self, audio_path, start_time, end_time, max_duration=10):
        if start_time >= end_time or start_time < 0:
            return None

        audio = AudioSegment.from_file(audio_path)
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        if start_ms >= len(audio) or end_ms <= start_ms:
            return None

        segment = audio[start_ms:end_ms]

        if len(segment) == 0:
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            segment.export(temp_file.name, format="wav")
            temp_path = temp_file.name

        waveform, sr = torchaudio.load(temp_path)

        if waveform.numel() == 0:
            os.unlink(temp_path)
            return None

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        max_samples = int(max_duration * self.target_sr)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < int(0.5 * self.target_sr):
            padding = int(0.5 * self.target_sr) - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        os.unlink(temp_path)
        return waveform.to(self.device)

    def extract_speaker_embedding_from_segment(self, audio_path, start_time, end_time):
        waveform = self.extract_audio_segment(audio_path, start_time, end_time)
        if waveform is None:
            return None

        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)
            return embedding.squeeze().cpu().numpy()

    def extract_speaker_embeddings_from_unified_data(
        self, unified_data, max_files_per_speaker=None
    ):
        original_audio_path = unified_data["metadata"]["audio_file"]

        if not os.path.exists(original_audio_path):
            print(f"Original audio file not found: {original_audio_path}")
            return None, None

        turns = unified_data["diarization"]["turns"]

        speaker_embeddings = defaultdict(list)
        speaker_counts = defaultdict(int)

        print("Extracting speaker embeddings from original audio...")
        for turn in turns:
            speaker = turn["speaker"]

            if (
                max_files_per_speaker
                and speaker_counts[speaker] >= max_files_per_speaker
            ):
                continue

            speaker_counts[speaker] += 1
            embedding = self.extract_speaker_embedding_from_segment(
                original_audio_path, turn["start_time"], turn["end_time"]
            )
            if embedding is not None:
                speaker_embeddings[speaker].append(embedding)
                if max_files_per_speaker:
                    print(
                        f"Processed {speaker}: Turn {turn['turn_id']} ({speaker_counts[speaker]}/{max_files_per_speaker})"
                    )
                else:
                    print(
                        f"Processed {speaker}: Turn {turn['turn_id']} ({speaker_counts[speaker]})"
                    )
            else:
                if max_files_per_speaker:
                    print(
                        f"Skipped {speaker}: Turn {turn['turn_id']} (failed extraction) ({speaker_counts[speaker]}/{max_files_per_speaker})"
                    )
                else:
                    print(
                        f"Skipped {speaker}: Turn {turn['turn_id']} (failed extraction) ({speaker_counts[speaker]})"
                    )

        speaker_mean_embeddings = {}
        for speaker, embeddings in speaker_embeddings.items():
            if embeddings:
                speaker_mean_embeddings[speaker] = np.mean(embeddings, axis=0)
                print(
                    f"Speaker {speaker}: {len(embeddings)} segments processed, embedding dim: {len(speaker_mean_embeddings[speaker])}"
                )

        return speaker_mean_embeddings

    def extract_event_embeddings_from_unified_data(self, unified_data):
        original_audio_path = unified_data["metadata"]["audio_file"]

        if not os.path.exists(original_audio_path):
            print(f"Original audio file not found: {original_audio_path}")
            return None, None

        paralinguistic_events = unified_data["audio_events"]["paralinguistic_events"]

        event_embeddings = []
        event_info = []

        print("Extracting event embeddings from original audio...")
        for event in paralinguistic_events:
            embedding = self.extract_speaker_embedding_from_segment(
                original_audio_path, event["start_time"], event["end_time"]
            )
            if embedding is not None:
                event_embeddings.append(embedding)
                event_info.append(
                    {
                        "event_id": event["event_id"],
                        "text": event["text"],
                        "start_time": event["start_time"],
                        "end_time": event["end_time"],
                        "duration": event["duration"],
                        "occurrence_count": event["occurrence_count"],
                    }
                )
                print(
                    f"Processed paralinguistic event {event['event_id']}: {event['text']}"
                )

        return np.array(event_embeddings) if event_embeddings else None, event_info

    def assign_events_clustering(
        self, speaker_embeddings, event_embeddings, event_info
    ):
        if len(speaker_embeddings) == 0 or event_embeddings is None:
            return []

        speaker_names = list(speaker_embeddings.keys())
        num_speakers = len(speaker_names)

        tag_groups = defaultdict(list)
        for i, event in enumerate(event_info):
            tag = event["text"]
            tag_groups[tag].append((i, event))

        print(f"Found {len(tag_groups)} unique paralinguistic tags:")
        for tag, events in tag_groups.items():
            print(f"  - {tag}: {len(events)} events")

        all_results = []

        for tag, tag_events in tag_groups.items():
            print(f"\nProcessing tag: '{tag}' ({len(tag_events)} events)")

            tag_indices = [idx for idx, _ in tag_events]
            tag_embeddings = event_embeddings[tag_indices]
            tag_event_info = [event for _, event in tag_events]

            tag_num_clusters = min(len(tag_events), num_speakers)

            if len(tag_events) == 1:
                scaler = StandardScaler()
                normalized_event = scaler.fit_transform([tag_embeddings[0]])
                speaker_matrix = np.array(
                    [speaker_embeddings[name] for name in speaker_names]
                )
                normalized_speakers = scaler.transform(speaker_matrix)
                similarities = cosine_similarity(normalized_event, normalized_speakers)[
                    0
                ]
                best_speaker_idx = np.argmax(similarities)
                assigned_speaker = speaker_names[best_speaker_idx]

                event_result = tag_event_info[0].copy()
                event_result["assigned_speaker"] = assigned_speaker
                event_result["cluster_id"] = f"{tag}_0"
                event_result["similarity_score"] = float(similarities[best_speaker_idx])
                event_result["tag_group"] = tag

                all_results.append(event_result)
                print(
                    f"  Single event -> {assigned_speaker} (sim: {similarities[best_speaker_idx]:.3f})"
                )
                continue

            print(
                f"  Using Agglomerative Clustering with {tag_num_clusters} clusters..."
            )

            scaler = StandardScaler()
            normalized_tag_events = scaler.fit_transform(tag_embeddings)

            clustering = AgglomerativeClustering(
                n_clusters=tag_num_clusters, linkage="ward", metric="euclidean"
            )
            cluster_labels = clustering.fit_predict(normalized_tag_events)

            cluster_centers = []
            for cluster_id in range(tag_num_clusters):
                cluster_points = normalized_tag_events[cluster_labels == cluster_id]
                if len(cluster_points) > 0:
                    cluster_center = np.mean(cluster_points, axis=0)
                    cluster_centers.append(cluster_center)
                else:
                    cluster_centers.append(normalized_tag_events[0])

            cluster_centers = np.array(cluster_centers)

            speaker_matrix = np.array(
                [speaker_embeddings[name] for name in speaker_names]
            )
            normalized_speakers = scaler.transform(speaker_matrix)

            similarities = cosine_similarity(normalized_speakers, cluster_centers)

            cluster_to_speaker = {}
            used_speakers = set()

            for cluster_idx in range(tag_num_clusters):
                best_speaker_idx = -1
                best_similarity = -1

                for speaker_idx, similarity in enumerate(similarities[:, cluster_idx]):
                    if (
                        speaker_names[speaker_idx] not in used_speakers
                        and similarity > best_similarity
                    ):
                        best_similarity = similarity
                        best_speaker_idx = speaker_idx

                if best_speaker_idx != -1:
                    speaker = speaker_names[best_speaker_idx]
                    cluster_to_speaker[cluster_idx] = speaker
                    used_speakers.add(speaker)
                    print(
                        f"  Cluster {cluster_idx} -> {speaker} (similarity: {best_similarity:.3f})"
                    )

            for cluster_idx in range(tag_num_clusters):
                if cluster_idx not in cluster_to_speaker:
                    remaining_speakers = [
                        s for s in speaker_names if s not in used_speakers
                    ]
                    if remaining_speakers:
                        cluster_to_speaker[cluster_idx] = remaining_speakers[0]
                        used_speakers.add(remaining_speakers[0])
                    else:
                        cluster_to_speaker[cluster_idx] = speaker_names[
                            cluster_idx % len(speaker_names)
                        ]

            for i, (event, label) in enumerate(zip(tag_event_info, cluster_labels)):
                assigned_speaker = cluster_to_speaker[label]

                event_result = event.copy()
                event_result["assigned_speaker"] = assigned_speaker
                event_result["cluster_id"] = f"{tag}_{label}"
                event_result["tag_group"] = tag

                speaker_emb = speaker_embeddings[assigned_speaker]
                event_emb = tag_embeddings[i]
                similarity = cosine_similarity([event_emb], [speaker_emb])[0][0]
                event_result["similarity_score"] = float(similarity)

                all_results.append(event_result)
                print(
                    f"  Event {event['event_id']} -> {assigned_speaker} (sim: {similarity:.3f})"
                )

        all_results.sort(key=lambda x: x["event_id"])

        print(f"\n=== TAG-BASED ASSIGNMENT SUMMARY ===")
        tag_speaker_counts = defaultdict(lambda: defaultdict(int))
        for result in all_results:
            tag_speaker_counts[result["tag_group"]][result["assigned_speaker"]] += 1

        for tag, speaker_counts in tag_speaker_counts.items():
            print(f"{tag}:")
            for speaker, count in speaker_counts.items():
                print(f"  {speaker}: {count} events")

        overall_speaker_counts = defaultdict(int)
        for result in all_results:
            overall_speaker_counts[result["assigned_speaker"]] += 1

        print(f"\nOVERALL SPEAKER ASSIGNMENT:")
        for speaker in speaker_names:
            count = overall_speaker_counts[speaker]
            print(f"{speaker}: {count} events assigned")

        return all_results

    def generate_enhanced_diarized_text(self, unified_data, assignment_results):
        words = unified_data["transcription"]["words"]

        assignment_map = {}
        for assignment in assignment_results:
            key = (assignment["start_time"], assignment["end_time"], assignment["text"])
            assignment_map[key] = assignment["assigned_speaker"]

        diarized_text_parts = []
        current_speaker = None
        line_buffer = ""

        for word in words:
            if word.get("type") == "word":
                speaker = word.get("speaker_id", "Unknown")

                if word.get("text"):
                    if speaker != current_speaker:
                        if line_buffer:
                            diarized_text_parts.append(line_buffer.strip())
                        line_buffer = f"{speaker}: {word['text']}"
                        current_speaker = speaker
                    else:
                        if line_buffer:
                            line_buffer += f" {word['text']}"
                        else:
                            line_buffer = f"{speaker}: {word['text']}"

            elif word.get("type") == "audio_event":
                event_key = (
                    word.get("start", 0),
                    word.get("end", 0),
                    word.get("text", ""),
                )
                assigned_speaker = assignment_map.get(event_key)

                if assigned_speaker:
                    event_text = word["text"]

                    if assigned_speaker == current_speaker and line_buffer:
                        line_buffer += f" {event_text}"
                    else:
                        if line_buffer:
                            diarized_text_parts.append(line_buffer.strip())
                        line_buffer = f"{assigned_speaker}: {event_text}"
                        current_speaker = assigned_speaker

        if line_buffer:
            diarized_text_parts.append(line_buffer.strip())

        return "\n".join(diarized_text_parts)

    def generate_turns_data(self, unified_data, assignment_results):
        words = unified_data["transcription"]["words"]

        assignment_map = {}
        for assignment in assignment_results:
            key = (assignment["start_time"], assignment["end_time"], assignment["text"])
            assignment_map[key] = assignment["assigned_speaker"]

        turns = []
        current_speaker = None
        current_text = ""
        current_start = None
        current_end = None

        for word in words:
            if word.get("type") == "word":
                speaker = word.get("speaker_id", "Unknown")
                text = word.get("text", "")
                start = word.get("start", 0)
                end = word.get("end", 0)

                if speaker != current_speaker:
                    if current_text and current_speaker:
                        turns.append(
                            {
                                "text": current_text.strip(),
                                "speaker": current_speaker,
                                "start_time": current_start,
                                "end_time": current_end,
                            }
                        )
                    current_speaker = speaker
                    current_text = text
                    current_start = start
                    current_end = end
                else:
                    current_text += f" {text}"
                    current_end = end

            elif word.get("type") == "audio_event":
                event_key = (
                    word.get("start", 0),
                    word.get("end", 0),
                    word.get("text", ""),
                )
                assigned_speaker = assignment_map.get(event_key)
                event_text = word.get("text", "audio_event")
                start = word.get("start", 0)
                end = word.get("end", 0)

                if assigned_speaker:
                    if assigned_speaker == current_speaker and current_text:
                        current_text += f" {event_text}"
                        current_end = end
                    else:
                        if current_text and current_speaker:
                            turns.append(
                                {
                                    "text": current_text.strip(),
                                    "speaker": current_speaker,
                                    "start_time": current_start,
                                    "end_time": current_end,
                                }
                            )
                        current_speaker = assigned_speaker
                        current_text = event_text
                        current_start = start
                        current_end = end

        if current_text and current_speaker:
            turns.append(
                {
                    "text": current_text.strip(),
                    "speaker": current_speaker,
                    "start_time": current_start,
                    "end_time": current_end,
                }
            )

        return turns

    def process_unified_data(self, unified_data_path):
        print(f"\n=== PROCESSING UNIFIED DATA with {self.model_type.upper()} ===")
        print(f"Unified data file: {unified_data_path}")

        if not os.path.exists(unified_data_path):
            print(f"Unified data file not found: {unified_data_path}")
            return None

        with open(unified_data_path, "r", encoding="utf-8") as f:
            unified_data = json.load(f)

        paralinguistic_events = unified_data["audio_events"]["paralinguistic_events"]

        if not paralinguistic_events:
            print(
                "No paralinguistic events found - skipping speaker embedding extraction"
            )
            assignments = []
            speaker_embeddings = {}
        else:
            speaker_embeddings = self.extract_speaker_embeddings_from_unified_data(
                unified_data
            )
            if not speaker_embeddings:
                print("No speaker embeddings found")
                return None

            (
                event_embeddings,
                event_info,
            ) = self.extract_event_embeddings_from_unified_data(unified_data)
            if event_embeddings is None or len(event_embeddings) == 0:
                print("No paralinguistic event embeddings found")
                assignments = []
            else:
                assignments = self.assign_events_clustering(
                    speaker_embeddings, event_embeddings, event_info
                )

        turns_data = self.generate_turns_data(unified_data, assignments)

        base_name = os.path.splitext(
            os.path.basename(unified_data["metadata"]["audio_file"])
        )[0]
        output_dir = "final_results"
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "metadata": {
                "model_type": self.model_type,
                "original_audio_file": unified_data["metadata"]["audio_file"],
                "unified_data_source": unified_data_path,
                "processing_timestamp": __import__("datetime")
                .datetime.now()
                .isoformat(),
            },
            "speakers": list(speaker_embeddings.keys()) if speaker_embeddings else [],
            "assignment_summary": {
                "total_events": len(assignments),
                "paralinguistic_events_assigned": len(assignments),
                "unique_tag_types": len(set([a["tag_group"] for a in assignments]))
                if assignments
                else 0,
            },
            "turns": turns_data,
        }

        output_file = os.path.join(output_dir, f"{base_name}_{self.model_type}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n=== ASSIGNMENT COMPLETED ===")
        print(f"Model: {self.model_type.upper()}")
        print(f"Processed {len(assignments)} events")
        print(f"Generated {len(turns_data)} turns")
        print(f"Results saved to: {output_file}")

        return {
            "output_file": output_file,
            "results": results,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Efficient speaker assignment using unified analysis data"
    )
    parser.add_argument(
        "unified_data_path", help="Path to the unified_analysis.json file"
    )
    args = parser.parse_args()

    assigner = SpeakerEmbeddingAssignerEfficient(model_type=args.model)
    result = assigner.process_unified_data(args.unified_data_path)

    if result:
        print(f"\n=== EFFICIENT ASSIGNMENT COMPLETED ===")
        print(f"Model: {args.model.upper()}")
        print(f"Results: {result['output_file']}")


def run_efficient_speaker_assignment(unified_data_path, model_type="ecapa"):
    print(f"\n=== RUNNING EFFICIENT SPEAKER ASSIGNMENT ===")
    print(f"Unified data: {unified_data_path}")
    print(f"Model: {model_type.upper()}")

    assigner = SpeakerEmbeddingAssignerEfficient(model_type=model_type)
    result = assigner.process_unified_data(unified_data_path)

    if result:
        print(f"\n=== EFFICIENT ASSIGNMENT COMPLETED ===")
        print(f"Results file: {result['output_file']}")

        return {
            "success": True,
            "output_file": result["output_file"],
            "full_result": result,
        }
    else:
        print("Failed to process speaker assignments")
        return {"success": False, "error": "Processing failed"}


if __name__ == "__main__":
    main()
