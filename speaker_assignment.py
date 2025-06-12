import os
import json
import numpy as np
import torch
import torchaudio
import librosa
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import argparse
from collections import defaultdict
from pathlib import Path
from speechbrain.inference import EncoderClassifier
from transformers import Wav2Vec2FeatureExtractor, WavLMModel


class SpeakerEmbeddingAssigner:
    def __init__(self, model_type="ecapa", device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_type = model_type
        print(f"Using device: {self.device}")
        print(f"Using model type: {model_type}")

        if model_type == "ecapa":
            self._init_ecapa_model()
        elif model_type == "wavlm":
            self._init_wavlm_model()
        else:
            raise ValueError("model_type must be 'ecapa' or 'wavlm'")

    # ECAPA-TDNN model is better than WavLM-based model
    def _init_ecapa_model(self):
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )
        self.target_sr = 16000
        print("ECAPA-TDNN model loaded successfully")

    def _init_wavlm_model(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus-sv").to(
            self.device
        )
        self.model.eval()
        self.target_sr = self.feature_extractor.sampling_rate
        print("WavLM-based model loaded successfully")

    def load_and_preprocess_audio(self, audio_path, max_duration=10):
        waveform, sr = torchaudio.load(audio_path)

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

        return waveform.to(self.device)

    def extract_speaker_embedding(self, audio_path):
        waveform = self.load_and_preprocess_audio(audio_path)
        if waveform is None:
            return None

        with torch.no_grad():
            if self.model_type == "ecapa":
                embedding = self.model.encode_batch(waveform)
                return embedding.squeeze().cpu().numpy()

            elif self.model_type == "wavlm":
                inputs = self.feature_extractor(
                    waveform.squeeze().cpu().numpy(),
                    sampling_rate=self.target_sr,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)
                embedding = torch.mean(outputs.last_hidden_state, dim=1)
                return embedding.squeeze().cpu().numpy()

    def extract_speaker_embeddings(self, results_dir, max_files_per_speaker=20):
        turns_info_path = os.path.join(results_dir, "turns_info.json")

        if not os.path.exists(turns_info_path):
            print(f"turns_info.json not found in {results_dir}")
            return None

        with open(turns_info_path, "r", encoding="utf-8") as f:
            turns_info = json.load(f)

        speaker_embeddings = defaultdict(list)
        speaker_counts = defaultdict(int)

        print("Extracting speaker embeddings...")
        for turn in turns_info:
            speaker = turn["speaker"]
            audio_file = turn["audio_file"]

            if speaker_counts[speaker] >= max_files_per_speaker:
                continue

            if os.path.exists(audio_file):
                embedding = self.extract_speaker_embedding(audio_file)
                if embedding is not None:
                    speaker_embeddings[speaker].append(embedding)
                    speaker_counts[speaker] += 1
                    print(
                        f"Processed {speaker}: {os.path.basename(audio_file)} ({speaker_counts[speaker]}/{max_files_per_speaker})"
                    )

        speaker_mean_embeddings = {}
        for speaker, embeddings in speaker_embeddings.items():
            if embeddings:
                speaker_mean_embeddings[speaker] = np.mean(embeddings, axis=0)
                print(
                    f"Speaker {speaker}: {len(embeddings)} files processed, embedding dim: {len(speaker_mean_embeddings[speaker])}"
                )

        return speaker_mean_embeddings, speaker_embeddings

    def extract_event_embeddings(self, results_dir):
        events_audio_info_path = os.path.join(results_dir, "events_audio_info.json")
        paralinguistic_events_path = os.path.join(
            results_dir, "paralinguistic_events.json"
        )

        if not os.path.exists(events_audio_info_path):
            print(f"events_audio_info.json not found in {results_dir}")
            return None, None

        if not os.path.exists(paralinguistic_events_path):
            print(f"paralinguistic_events.json not found in {results_dir}")
            return None, None

        with open(events_audio_info_path, "r", encoding="utf-8") as f:
            events_audio_info = json.load(f)

        with open(paralinguistic_events_path, "r", encoding="utf-8") as f:
            paralinguistic_events = json.load(f)

        # Create a mapping from (start_time, end_time, text) to paralinguistic event
        para_event_map = {}
        for para_event in paralinguistic_events:
            key = (para_event["start_time"], para_event["end_time"], para_event["text"])
            para_event_map[key] = para_event

        event_embeddings = []
        event_info = []

        print("Extracting event embeddings for paralinguistic events only...")
        for event_audio in events_audio_info:
            # Check if this event is in paralinguistic events
            key = (
                event_audio["start_time"],
                event_audio["end_time"],
                event_audio["text"],
            )
            if key in para_event_map:
                para_event = para_event_map[key]
                audio_file = event_audio["audio_file"]

                if os.path.exists(audio_file):
                    embedding = self.extract_speaker_embedding(audio_file)
                    if embedding is not None:
                        event_embeddings.append(embedding)
                        event_info.append(
                            {
                                "event_id": event_audio["event_id"],
                                "text": event_audio["text"],
                                "start_time": event_audio["start_time"],
                                "end_time": event_audio["end_time"],
                                "duration": event_audio["duration"],
                                "audio_file": audio_file,
                                "paralinguistic_score": para_event.get(
                                    "paralinguistic_score", 0
                                ),
                            }
                        )
                        print(
                            f"Processed paralinguistic event {event_audio['event_id']}: {event_audio['text']}"
                        )
                else:
                    print(f"Audio file not found: {audio_file}")
            else:
                print(
                    f"Skipping non-paralinguistic event {event_audio['event_id']}: {event_audio['text']}"
                )

        return np.array(event_embeddings) if event_embeddings else None, event_info

    def assign_events_clustering(
        self, speaker_embeddings, event_embeddings, event_info
    ):
        if len(speaker_embeddings) == 0 or event_embeddings is None:
            return []

        speaker_names = list(speaker_embeddings.keys())
        num_speakers = len(speaker_names)

        # Group events by tag (text)
        tag_groups = defaultdict(list)
        for i, event in enumerate(event_info):
            tag = event["text"]
            tag_groups[tag].append((i, event))

        print(f"Found {len(tag_groups)} unique paralinguistic tags:")
        for tag, events in tag_groups.items():
            print(f"  - {tag}: {len(events)} events")

        all_results = []

        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler

        # Process each tag group separately
        for tag, tag_events in tag_groups.items():
            print(f"\nProcessing tag: '{tag}' ({len(tag_events)} events)")

            # Extract embeddings for this tag group
            tag_indices = [idx for idx, _ in tag_events]
            tag_embeddings = event_embeddings[tag_indices]
            tag_event_info = [event for _, event in tag_events]

            # Determine number of clusters for this tag
            # Use minimum of: number of events in this tag, number of speakers
            tag_num_clusters = min(len(tag_events), num_speakers)

            if len(tag_events) == 1:
                # Only one event for this tag - assign to most similar speaker
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

            # Multiple events for this tag - use clustering
            print(
                f"  Using Agglomerative Clustering with {tag_num_clusters} clusters..."
            )

            scaler = StandardScaler()
            normalized_tag_events = scaler.fit_transform(tag_embeddings)

            clustering = AgglomerativeClustering(
                n_clusters=tag_num_clusters, linkage="ward", metric="euclidean"
            )
            cluster_labels = clustering.fit_predict(normalized_tag_events)

            # Calculate cluster centers
            cluster_centers = []
            for cluster_id in range(tag_num_clusters):
                cluster_points = normalized_tag_events[cluster_labels == cluster_id]
                if len(cluster_points) > 0:
                    cluster_center = np.mean(cluster_points, axis=0)
                    cluster_centers.append(cluster_center)
                else:
                    cluster_centers.append(normalized_tag_events[0])

            cluster_centers = np.array(cluster_centers)

            # Map clusters to speakers
            speaker_matrix = np.array(
                [speaker_embeddings[name] for name in speaker_names]
            )
            normalized_speakers = scaler.transform(speaker_matrix)

            similarities = cosine_similarity(normalized_speakers, cluster_centers)

            cluster_to_speaker = {}
            used_speakers = set()

            # Assign speakers to clusters based on highest similarity
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

            # Handle unassigned clusters
            for cluster_idx in range(tag_num_clusters):
                if cluster_idx not in cluster_to_speaker:
                    remaining_speakers = [
                        s for s in speaker_names if s not in used_speakers
                    ]
                    if remaining_speakers:
                        cluster_to_speaker[cluster_idx] = remaining_speakers[0]
                        used_speakers.add(remaining_speakers[0])
                    else:
                        # Reuse speakers if all are assigned
                        cluster_to_speaker[cluster_idx] = speaker_names[
                            cluster_idx % len(speaker_names)
                        ]

            # Assign events to speakers
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

        # Sort results by event_id for consistent output
        all_results.sort(key=lambda x: x["event_id"])

        # Print summary statistics
        print(f"\n=== TAG-BASED ASSIGNMENT SUMMARY ===")
        tag_speaker_counts = defaultdict(lambda: defaultdict(int))
        for result in all_results:
            tag_speaker_counts[result["tag_group"]][result["assigned_speaker"]] += 1

        for tag, speaker_counts in tag_speaker_counts.items():
            print(f"{tag}:")
            for speaker, count in speaker_counts.items():
                print(f"  {speaker}: {count} events")

        # Overall speaker counts
        overall_speaker_counts = defaultdict(int)
        for result in all_results:
            overall_speaker_counts[result["assigned_speaker"]] += 1

        print(f"\nOVERALL SPEAKER ASSIGNMENT:")
        for speaker in speaker_names:
            count = overall_speaker_counts[speaker]
            print(f"{speaker}: {count} events assigned")

        return all_results

    def process_directory(self, results_dir):
        print(f"\n=== PROCESSING {results_dir} with {self.model_type.upper()} ===")

        if not os.path.exists(results_dir):
            print(f"Directory not found: {results_dir}")
            return None

        speaker_embeddings, speaker_all_embeddings = self.extract_speaker_embeddings(
            results_dir
        )
        if not speaker_embeddings:
            print("No speaker embeddings found")
            return None

        event_embeddings, event_info = self.extract_event_embeddings(results_dir)
        if event_embeddings is None or len(event_embeddings) == 0:
            print("No paralinguistic event embeddings found")
            return None

        results = self.assign_events_clustering(
            speaker_embeddings, event_embeddings, event_info
        )

        output_file = os.path.join(
            results_dir,
            f"paralinguistic_speaker_assignments_{self.model_type}.json",
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": self.model_type,
                    "speakers": list(speaker_embeddings.keys()),
                    "total_events": len(results),
                    "assignments": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        summary_file = os.path.join(
            results_dir, f"assignment_summary_{self.model_type}.txt"
        )

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"SPEAKER ASSIGNMENT USING {self.model_type.upper()}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Model: {self.model_type.upper()}\n")
            f.write(f"Speakers: {', '.join(speaker_embeddings.keys())}\n")
            f.write(f"Total Events: {len(results)}\n\n")

            # Group results by tag for organized display
            tag_groups = defaultdict(list)
            for result in results:
                tag_groups[result.get("tag_group", result["text"])].append(result)

            f.write("EVENTS BY TAG GROUP:\n")
            f.write("-" * 30 + "\n")
            for tag, tag_results in tag_groups.items():
                f.write(f"\n{tag.upper()} ({len(tag_results)} events):\n")
                for result in tag_results:
                    f.write(
                        f"  Event {result['event_id']:02d}: {result['assigned_speaker']} "
                        f"(sim: {result['similarity_score']:.3f}) [{result['start_time']:.1f}s-{result['end_time']:.1f}s]\n"
                    )

            f.write(f"\nALL EVENTS CHRONOLOGICALLY:\n")
            f.write("-" * 30 + "\n")
            for result in results:
                f.write(
                    f"Event {result['event_id']:02d}: {result['text']} -> {result['assigned_speaker']} "
                    f"(sim: {result['similarity_score']:.3f}) [{result['start_time']:.1f}s-{result['end_time']:.1f}s]\n"
                )

            # Tag-based speaker counts
            tag_speaker_counts = defaultdict(lambda: defaultdict(int))
            for result in results:
                tag_speaker_counts[result.get("tag_group", result["text"])][
                    result["assigned_speaker"]
                ] += 1

            f.write(f"\nEVENTS PER SPEAKER BY TAG:\n")
            f.write("-" * 30 + "\n")
            for tag, speaker_counts in tag_speaker_counts.items():
                f.write(f"{tag}:\n")
                for speaker, count in speaker_counts.items():
                    f.write(f"  {speaker}: {count} events\n")

            # Overall speaker counts
            speaker_counts = defaultdict(int)
            for result in results:
                speaker_counts[result["assigned_speaker"]] += 1

            f.write(f"\nOVERALL EVENTS PER SPEAKER:\n")
            f.write("-" * 30 + "\n")
            for speaker, count in speaker_counts.items():
                f.write(f"{speaker}: {count} events\n")

        print(f"Results saved to: {output_file}")
        print(f"Summary saved to: {summary_file}")

        # Generate enhanced diarized text with speaker assignments
        assignment_results = {
            "model_type": self.model_type,
            "speakers": list(speaker_embeddings.keys()),
            "total_events": len(results),
            "assignments": results,
        }

        diarized_result = self.generate_enhanced_diarized_text(
            results_dir, assignment_results
        )

        return {
            "output_file": output_file,
            "summary_file": summary_file,
            "assignments": results,
            "enhanced_diarized": diarized_result,
        }

    def generate_enhanced_diarized_text(self, results_dir, assignment_results):
        """Generate enhanced diarized.txt with embedding-based speaker assignments and paralinguistic events"""

        # Load original transcription data
        transcription_file = os.path.join(results_dir, "transcription.json")
        if not os.path.exists(transcription_file):
            print(f"transcription.json not found in {results_dir}")
            return None

        with open(transcription_file, "r", encoding="utf-8") as f:
            transcription_data = json.load(f)

        if not transcription_data.get("words"):
            print("No word-level data found in transcription")
            return None

        # Create assignment mapping from start/end times
        assignment_map = {}
        for assignment in assignment_results["assignments"]:
            key = (assignment["start_time"], assignment["end_time"], assignment["text"])
            assignment_map[key] = assignment["assigned_speaker"]

        # Generate enhanced diarized text
        diarized_text_parts = []
        current_speaker = None
        line_buffer = ""

        for word in transcription_data["words"]:
            if word.get("type") == "word":
                speaker = getattr(word, "speaker_id", word.get("speaker_id", "Unknown"))

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
                # Check if this event has a speaker assignment (paralinguistic)
                event_key = (
                    word.get("start", 0),
                    word.get("end", 0),
                    word.get("text", ""),
                )
                assigned_speaker = assignment_map.get(event_key)

                if assigned_speaker:
                    # Paralinguistic event with assigned speaker
                    event_text = word["text"]

                    if assigned_speaker == current_speaker and line_buffer:
                        # Same speaker - add to current line
                        line_buffer += f" {event_text}"
                    else:
                        # Different speaker or no current line - start new line
                        if line_buffer:
                            diarized_text_parts.append(line_buffer.strip())
                        line_buffer = f"{assigned_speaker}: {event_text}"
                        current_speaker = assigned_speaker
                # Non-paralinguistic events are completely skipped (not included in output)

        # Add final line if exists
        if line_buffer:
            diarized_text_parts.append(line_buffer.strip())

        # Save enhanced diarized text
        enhanced_diarized_file = os.path.join(
            results_dir, f"@diarized_{self.model_type}.txt"
        )
        with open(enhanced_diarized_file, "w", encoding="utf-8") as f:
            f.write("\n".join(diarized_text_parts))

        print(f"Enhanced diarized text saved to: {enhanced_diarized_file}")

        # Generate statistics
        total_paralinguistic_events = len(assignment_results["assignments"])

        # Count paralinguistic events in text (marked with [])
        paralinguistic_count = 0
        for line in diarized_text_parts:
            if ": " in line:
                content = line.split(": ", 1)[1] if ": " in line else line
                paralinguistic_count += content.count("[")

        stats = {
            "total_lines": len(diarized_text_parts),
            "speaker_lines": len(
                diarized_text_parts
            ),  # All lines are speaker lines now
            "total_paralinguistic_events": total_paralinguistic_events,
            "paralinguistic_events_in_text": paralinguistic_count,
            "non_paralinguistic_events_excluded": True,
        }

        # Save statistics
        stats_file = os.path.join(
            results_dir, f"diarization_stats_{self.model_type}.json"
        )
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Diarization statistics:")
        print(f"- Total lines: {stats['total_lines']}")
        print(f"- Speaker lines: {stats['speaker_lines']}")
        print(
            f"- Total paralinguistic events processed: {stats['total_paralinguistic_events']}"
        )
        print(
            f"- Paralinguistic events in text: {stats['paralinguistic_events_in_text']}"
        )
        print(
            f"- Non-paralinguistic events excluded: {stats['non_paralinguistic_events_excluded']}"
        )

        return {
            "enhanced_diarized_file": enhanced_diarized_file,
            "stats_file": stats_file,
            "stats": stats,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Speaker assignment using ECAPA-TDNN or WavLM models"
    )
    parser.add_argument("results_dir", help="Path to the elevenlabs results directory")
    parser.add_argument(
        "--model",
        choices=["ecapa", "wavlm"],
        default="ecapa",
        help="Model type: ecapa (ECAPA-TDNN) or wavlm (WavLM-based)",
    )

    args = parser.parse_args()

    assigner = SpeakerEmbeddingAssigner(model_type=args.model)
    result = assigner.process_directory(args.results_dir)

    if result:
        print(f"\n=== ASSIGNMENT COMPLETED ===")
        print(f"Model: {args.model.upper()}")
        print(f"Processed {len(result['assignments'])} events")
        print(f"Results: {result['output_file']}")
        print(f"Summary: {result['summary_file']}")

        if result.get("enhanced_diarized"):
            print(
                f"Enhanced diarized text: {result['enhanced_diarized']['enhanced_diarized_file']}"
            )
            print(f"Diarization stats: {result['enhanced_diarized']['stats_file']}")


def run_enhanced_diarization(results_dir, model_type="ecapa"):
    """
    Run complete enhanced diarization pipeline

    Args:
        results_dir: Path to elevenlabs results directory
        model_type: "ecapa" or "wavlm"

    Returns:
        dict: Results including enhanced diarized text file path
    """
    print(f"\n=== RUNNING ENHANCED DIARIZATION PIPELINE ===")
    print(f"Directory: {results_dir}")
    print(f"Model: {model_type.upper()}")

    assigner = SpeakerEmbeddingAssigner(model_type=model_type)
    result = assigner.process_directory(results_dir)

    if result and result.get("enhanced_diarized"):
        enhanced_file = result["enhanced_diarized"]["enhanced_diarized_file"]
        stats = result["enhanced_diarized"]["stats"]

        print(f"\n=== ENHANCED DIARIZATION COMPLETED ===")
        print(f"Enhanced diarized file: {enhanced_file}")
        print(f"Statistics:")
        print(f"  - Total lines: {stats['total_lines']}")
        print(f"  - Speaker lines: {stats['speaker_lines']}")
        print(
            f"  - Total paralinguistic events processed: {stats['total_paralinguistic_events']}"
        )
        print(
            f"  - Paralinguistic events in text: {stats['paralinguistic_events_in_text']}"
        )
        print(
            f"  - Non-paralinguistic events excluded: {stats['non_paralinguistic_events_excluded']}"
        )

        return {
            "success": True,
            "enhanced_diarized_file": enhanced_file,
            "stats": stats,
            "full_result": result,
        }
    else:
        print("Failed to generate enhanced diarization")
        return {"success": False, "error": "Processing failed"}


if __name__ == "__main__":
    main()
