import os
import json
import io
from dotenv import load_dotenv
from pydub import AudioSegment
import openai
from elevenlabs import ElevenLabs
import torch
import torchaudio
from speechbrain.inference import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import tempfile
import numpy as np
import argparse

PARALINGUISTIC_EVENTS = [
    "laugh",
    "chuckle",
    "cry",
    "gasp",
    "sigh",
    "cough",
    "sneeze",
    "clear_throat",
    "breathe",
    "yawn",
    "snore",
    "grunt",
    "groan",
    "hiccup",
    "burp",
    "hum",
    "whistle",
    "whisper",
    "pant",
    "sniffle",
    "babble",
    "scream",
]

load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"), timeout=1000)
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def speech_to_text_with_diarization(audio_path):
    audio = AudioSegment.from_file(audio_path)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    result = client.speech_to_text.convert(
        file=buffer,
        model_id="scribe_v1_experimental",
        diarize=True,
        tag_audio_events=True,
    )
    return result


def extract_turns_from_words(words):
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
                            "start": turn_start,
                            "end": turn_end,
                            "text": turn_text,
                            "word_count": len(current_turn_words),
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
                "start": turn_start,
                "end": turn_end,
                "text": turn_text,
                "word_count": len(current_turn_words),
            }
        )
    return turns


def extract_audio_events(words):
    audio_events = []
    for word in words:
        if word.get("type") == "audio_event":
            audio_events.append(
                {
                    "text": word.get("text", ""),
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "duration": word.get("end", 0) - word.get("start", 0),
                    "type": "audio_event",
                }
            )
    audio_events.sort(key=lambda x: x["start"])
    return audio_events


def filter_paralinguistic_events(audio_events):
    cache_file = "tag_classification_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    def save_cache():
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write("{\n")
            for i, (k, v) in enumerate(cache.items()):
                line = f"  {json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
                if i < len(cache) - 1:
                    line += ","
                f.write(line + "\n")
            f.write("}\n")

    paralinguistic_events = []
    non_paralinguistic_events = []
    for idx, event in enumerate(audio_events):
        tag_name = event["text"]
        if tag_name in cache:
            classification, count = cache[tag_name][0], cache[tag_name][1]
            cache[tag_name][1] += 1
            save_cache()
        else:
            prompt = f"""Classify this Korean audio tag into one of these categories: {', '.join(PARALINGUISTIC_EVENTS)}\nAll categories are human paralinguistic sounds (including whistling, humming, etc). If the tag describes a human sound matching any category above, return that category name. If it's music, mechanical, or environmental, return 'None'.\nKorean audio tag: {tag_name}\nCategory:"""
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20,
            )
            result = response.choices[0].message.content.strip().lower()
            if result not in PARALINGUISTIC_EVENTS:
                result = "none"
            cache[tag_name] = [result, 1]
            save_cache()
            classification, count = result, 1
        event_with_score = event.copy()
        event_with_score["classification"] = classification
        event_with_score["occurrence_count"] = count
        if classification != "none":
            event_with_score["original_text"] = event_with_score["text"]
            event_with_score["text"] = f"<{classification}>"
            event_with_score["event_id"] = idx + 1
            if "start_time" in event_with_score:
                event_with_score["start"] = event_with_score.pop("start_time")
            if "end_time" in event_with_score:
                event_with_score["end"] = event_with_score.pop("end_time")
            paralinguistic_events.append(event_with_score)
        else:
            if "start_time" in event_with_score:
                event_with_score["start"] = event_with_score.pop("start_time")
            if "end_time" in event_with_score:
                event_with_score["end"] = event_with_score.pop("end_time")
            non_paralinguistic_events.append(event_with_score)
    return paralinguistic_events, non_paralinguistic_events


class SpeakerEmbeddingAssigner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )
        self.target_sr = 16000

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
            waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
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

    def extract_speaker_embeddings(self, audio_path, turns):
        speaker_embeddings = defaultdict(list)
        for turn in turns:
            speaker = turn["speaker"]
            embedding = self.extract_speaker_embedding_from_segment(
                audio_path, turn["start"], turn["end"]
            )
            if embedding is not None:
                speaker_embeddings[speaker].append(embedding)
        speaker_mean_embeddings = {}
        for speaker, embeddings in speaker_embeddings.items():
            if embeddings:
                speaker_mean_embeddings[speaker] = np.mean(embeddings, axis=0)
        return speaker_mean_embeddings

    def extract_event_embeddings(self, audio_path, paralinguistic_events):
        event_embeddings = []
        event_info = []
        for event in paralinguistic_events:
            event_text = event["text"]
            if not (event_text.startswith("<") and event_text.endswith(">")):
                continue
            embedding = self.extract_speaker_embedding_from_segment(
                audio_path, event["start"], event["end"]
            )
            if embedding is not None:
                event_embeddings.append(embedding)
                event_info.append(event)
        return np.array(event_embeddings) if event_embeddings else None, event_info

    def assign_events(self, speaker_embeddings, event_embeddings, event_info):
        if len(speaker_embeddings) == 0 or event_embeddings is None:
            return []
        speaker_names = list(speaker_embeddings.keys())
        num_speakers = len(speaker_names)
        tag_groups = defaultdict(list)
        for i, event in enumerate(event_info):
            tag = event["text"]
            tag_groups[tag].append((i, event))
        all_results = []
        for tag, tag_events in tag_groups.items():
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
                continue
            scaler = StandardScaler()
            normalized_tag_events = scaler.fit_transform(tag_embeddings)
            from sklearn.cluster import AgglomerativeClustering

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
        all_results.sort(key=lambda x: x["event_id"])
        return all_results

    def generate_turns(self, words, valid_classifications):
        turns = []
        current_speaker = None
        current_text = ""
        current_start = None
        current_end = None
        current_tag_count = 0
        for w in words:
            if w["type"] == "spacing":
                continue
            if w["type"] == "audio_event":
                if w["text"].strip("<>") not in valid_classifications:
                    continue
            speaker = w.get("speaker_id", "Unknown")
            text = w.get("text", "")
            start = w.get("start", 0)
            end = w.get("end", 0)
            if speaker != current_speaker:
                if current_text and current_speaker is not None:
                    turns.append(
                        {
                            "text": current_text.strip(),
                            "speaker": current_speaker,
                            "start": current_start,
                            "end": current_end,
                            "tag_count": current_tag_count,
                        }
                    )
                current_speaker = speaker
                current_text = text
                current_start = start
                current_end = end
                current_tag_count = 1 if w["type"] == "audio_event" else 0
            else:
                current_text += " " + text
                current_end = end
                if w["type"] == "audio_event":
                    current_tag_count += 1
        if current_text and current_speaker is not None:
            turns.append(
                {
                    "text": current_text.strip(),
                    "speaker": current_speaker,
                    "start": current_start,
                    "end": current_end,
                    "tag_count": current_tag_count,
                }
            )
        return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to the audio file")
    args = parser.parse_args()
    audio_path = args.audio_path
    # 1. ASR/speaker diarization/event tagging (ElevenLabs)
    result = speech_to_text_with_diarization(audio_path)
    if not result or not hasattr(result, "words") or not result.words:
        print("No transcription result or words found")
        exit(1)
    words = []
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
        words.append(word_dict)
    words.sort(key=lambda x: x.get("start", 0))
    print(f"Detected speakers: {len(speakers)} ({', '.join(str(s) for s in speakers)})")
    print("1. ASR/speaker diarization/event tagging (ElevenLabs) done!")
    # 2. turn/event extraction
    turns = extract_turns_from_words(words)
    audio_events = extract_audio_events(words)
    print("2. turn/event extraction done!")
    # 3. paralinguistic event classification (LLM)
    paralinguistic_events, _ = filter_paralinguistic_events(audio_events)
    print("3. paralinguistic event classification (LLM) done!")
    para_map = {
        (e["start"], e["end"], e["original_text"]): e["text"]
        for e in paralinguistic_events
        if "original_text" in e
    }
    for w in words:
        if w["type"] == "audio_event":
            key = (w["start"], w["end"], w["text"])
            if key in para_map:
                w["text"] = para_map[key]
    audio_events = [w for w in words if w["type"] == "audio_event"]
    # 4. speaker embedding/event embedding/speaker assign
    assigner = SpeakerEmbeddingAssigner()
    speaker_embeddings = assigner.extract_speaker_embeddings(audio_path, turns)
    event_embeddings, event_info = assigner.extract_event_embeddings(
        audio_path, paralinguistic_events
    )
    assignments = assigner.assign_events(
        speaker_embeddings, event_embeddings, event_info
    )
    print("5. speaker embedding/event embedding/speaker assign done!")
    # 5. final save
    valid_classifications = set(a["classification"] for a in assignments)
    print(valid_classifications)
    turns_data = assigner.generate_turns(words, valid_classifications)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = "final_results"
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "metadata": {
            "model_type": "ecapa",
            "original_audio_file": audio_path,
            "processing_timestamp": __import__("datetime").datetime.now().isoformat(),
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
    output_file = os.path.join(output_dir, f"{base_name}_ecapa.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("6. final save done!")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
