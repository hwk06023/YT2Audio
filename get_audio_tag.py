import os
import torch
import librosa
import numpy as np
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import json
from pathlib import Path
from datetime import timedelta
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler


class AudioTagger:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.model.to(self.device)
        self.model.eval()

        self.target_sample_rate = self.feature_extractor.sampling_rate

    def seconds_to_hms(self, seconds):
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def load_audio_segment(self, file_path, start_time, duration=5):
        audio, sr = librosa.load(
            file_path, sr=self.target_sample_rate, offset=start_time, duration=duration
        )

        target_length = int(duration * self.target_sample_rate)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        elif len(audio) > target_length:
            audio = audio[:target_length]

        return audio

    def predict_segment(self, audio_segment, multi_tag_threshold=0.2):
        inputs = self.feature_extractor(
            audio_segment, sampling_rate=self.target_sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        prob_array = probabilities[0].cpu().numpy()

        results = []
        for i, prob in enumerate(prob_array):
            if prob >= multi_tag_threshold:
                class_name = self.model.config.id2label[i]
                results.append({"tag_name": class_name, "probability": float(prob)})

        results.sort(key=lambda x: x["probability"], reverse=True)
        return results

    def detect_change_points(self, audio_path, window_size=5, hop_size=2.5):
        audio_info = librosa.get_duration(path=audio_path)
        total_duration = audio_info

        timestamps = []
        features = []
        current_time = 0

        print(f"Analyzing {Path(audio_path).name} for change points...")

        while current_time < total_duration:
            try:
                audio_segment = self.load_audio_segment(
                    audio_path, current_time, window_size
                )
                predictions = self.predict_segment(
                    audio_segment, multi_tag_threshold=0.1
                )

                if predictions:
                    feature_vector = [0.0] * 527  # AudioSet has 527 classes
                    for pred in predictions:
                        class_id = next(
                            k
                            for k, v in self.model.config.id2label.items()
                            if v == pred["tag_name"]
                        )
                        feature_vector[class_id] = pred["probability"]

                    timestamps.append(current_time)
                    features.append(feature_vector)

            except Exception as e:
                print(f"Error at {current_time:.1f}s: {str(e)}")

            current_time += hop_size

        if len(features) < 2:
            return [0, total_duration]

        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        distances = []
        for i in range(1, len(features_scaled)):
            dist = np.linalg.norm(features_scaled[i] - features_scaled[i - 1])
            distances.append(dist)

        distances = np.array(distances)
        threshold = np.mean(distances) + 1.5 * np.std(distances)
        peaks, _ = find_peaks(distances, height=threshold, distance=int(3 / hop_size))

        change_points = [0]
        for peak in peaks:
            change_points.append(timestamps[peak + 1])
        change_points.append(total_duration)

        return sorted(list(set(change_points)))

    def merge_similar_segments(self, segments, similarity_threshold=0.7):
        if not segments:
            return segments

        merged = [segments[0]]

        for current in segments[1:]:
            last = merged[-1]

            last_tags = set(tag["tag_name"] for tag in last["tags"])
            current_tags = set(tag["tag_name"] for tag in current["tags"])

            if last_tags and current_tags:
                intersection = len(last_tags & current_tags)
                union = len(last_tags | current_tags)
                similarity = intersection / union if union > 0 else 0

                if (
                    similarity >= similarity_threshold
                    and current["start_time_seconds"] - last["end_time_seconds"] <= 5
                ):
                    last["end_time"] = current["end_time"]
                    last["end_time_seconds"] = current["end_time_seconds"]

                    combined_tags = {}
                    for tag in last["tags"]:
                        combined_tags[tag["tag_name"]] = tag["probability"]
                    for tag in current["tags"]:
                        if tag["tag_name"] in combined_tags:
                            combined_tags[tag["tag_name"]] = max(
                                combined_tags[tag["tag_name"]], tag["probability"]
                            )
                        else:
                            combined_tags[tag["tag_name"]] = tag["probability"]

                    last["tags"] = [
                        {"tag_name": name, "probability": prob}
                        for name, prob in combined_tags.items()
                    ]
                    last["tags"].sort(key=lambda x: x["probability"], reverse=True)
                else:
                    merged.append(current)
            else:
                merged.append(current)

        return merged

    def process_file(self, audio_path, min_probability=0.2):
        change_points = self.detect_change_points(audio_path)

        results = []
        print(f"Found {len(change_points)-1} segments")

        for i in range(len(change_points) - 1):
            start_time = change_points[i]
            end_time = change_points[i + 1]
            duration = end_time - start_time

            if duration < 1:  # Skip very short segments
                continue

            try:
                audio_segment = self.load_audio_segment(
                    audio_path, start_time, min(duration, 10)
                )
                predictions = self.predict_segment(
                    audio_segment, multi_tag_threshold=min_probability
                )

                if predictions:
                    segment = {
                        "start_time": self.seconds_to_hms(start_time),
                        "end_time": self.seconds_to_hms(end_time),
                        "start_time_seconds": start_time,
                        "end_time_seconds": end_time,
                        "tags": predictions,
                    }
                    results.append(segment)

                    tag_names = ", ".join(
                        [
                            f"{p['tag_name']} ({p['probability']:.3f})"
                            for p in predictions[:3]
                        ]
                    )
                    print(
                        f"  {self.seconds_to_hms(start_time)}-{self.seconds_to_hms(end_time)}: {tag_names}"
                    )

            except Exception as e:
                print(
                    f"Error processing segment {self.seconds_to_hms(start_time)}-{self.seconds_to_hms(end_time)}: {str(e)}"
                )

        merged_results = self.merge_similar_segments(results)
        print(f"Merged to {len(merged_results)} segments")

        final_results = []
        for segment in merged_results:
            for tag in segment["tags"]:
                final_results.append(
                    {
                        "tag_name": tag["tag_name"],
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "probability": tag["probability"],
                    }
                )

        return final_results

    def process_directory(self, data_dir="data", output_dir="audio_tag_result"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        audio_files = list(Path(data_dir).glob("*.wav"))
        print(f"Found {len(audio_files)} wav files")

        for audio_file in audio_files:
            output_filename = output_path / f"{audio_file.stem}.json"

            if output_filename.exists():
                print(
                    f"Skipping {audio_file.name} - {output_filename.name} already exists"
                )
                continue

            try:
                results = self.process_file(str(audio_file))

                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                print(f"Saved {len(results)} tags to {output_filename}")
                print()

            except Exception as e:
                print(f"Error processing {audio_file.name}: {str(e)}")


def main():
    tagger = AudioTagger()
    tagger.process_directory()


if __name__ == "__main__":
    main()
