import os
import glob
import json
import torch
from transformers import (
    pipeline,
    ClapModel,
    ClapProcessor,
    ASTFeatureExtractor,
    ASTForAudioClassification,
)
import librosa
import numpy as np


def classify_with_clap_htsat_fused(audio_path):
    try:
        audio_classifier = pipeline(
            task="zero-shot-audio-classification", model="laion/clap-htsat-fused"
        )
        audio, sr = librosa.load(audio_path, sr=48000)
        result = audio_classifier(
            audio,
            candidate_labels=[
                "music",
                "speech",
                "sound effect",
                "noise",
                "silence",
                "background music",
                "vocal",
                "instrumental",
                "ambient sound",
                "applause",
                "laughter",
                "breathing",
                "footsteps",
            ],
        )
        return result[:3]
    except Exception as e:
        print(f"Error with CLAP model: {e}")
        return []


def classify_with_ast_audioset(audio_path):
    try:
        feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_ids = torch.topk(logits, 5).indices[0].tolist()
        predictions = []
        for class_id in predicted_class_ids:
            label = model.config.id2label[class_id]
            score = torch.softmax(logits, dim=-1)[0][class_id].item()
            predictions.append({"label": label, "score": score})

        return predictions
    except Exception as e:
        print(f"Error with AST model: {e}")
        return []


def classify_with_larger_clap_general(audio_path):
    try:
        model = ClapModel.from_pretrained("laion/larger_clap_general")
        processor = ClapProcessor.from_pretrained("laion/larger_clap_general")

        audio, sr = librosa.load(audio_path, sr=48000)
        inputs = processor(audios=audio, return_tensors="pt")

        text_candidates = [
            "this is music",
            "this is speech",
            "this is a sound effect",
            "this is background music",
            "this is singing",
            "this is instrumental music",
            "this is ambient sound",
            "this is noise",
            "this is silence",
        ]

        text_inputs = processor(text=text_candidates, return_tensors="pt", padding=True)

        with torch.no_grad():
            audio_features = model.get_audio_features(**inputs)
            text_features = model.get_text_features(**text_inputs)

            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * audio_features @ text_features.T).softmax(dim=-1)

        results = []
        for i, candidate in enumerate(text_candidates):
            score = similarity[0][i].item()
            results.append({"label": candidate, "score": score})

        return sorted(results, key=lambda x: x["score"], reverse=True)[:3]
    except Exception as e:
        print(f"Error with larger CLAP model: {e}")
        return []


def process_all_turns():
    base_dir = "turns"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return

    all_audio_files = glob.glob(f"{base_dir}/**/turn_*.wav", recursive=True)

    if not all_audio_files:
        print("No turn audio files found")
        return

    print(f"Found {len(all_audio_files)} audio files to process")

    for audio_file in all_audio_files:
        print(f"Processing: {audio_file}")

        base_name = os.path.splitext(audio_file)[0]

        clap_result = classify_with_clap_htsat_fused(audio_file)
        ast_result = classify_with_ast_audioset(audio_file)
        larger_clap_result = classify_with_larger_clap_general(audio_file)

        with open(f"{base_name}_clap_htsat.txt", "w", encoding="utf-8") as f:
            f.write("CLAP-HTSAT-FUSED Results:\n")
            for item in clap_result:
                f.write(f"{item['label']}: {item['score']:.4f}\n")

        with open(f"{base_name}_ast_audioset.txt", "w", encoding="utf-8") as f:
            f.write("AST-AudioSet Results:\n")
            for item in ast_result:
                f.write(f"{item['label']}: {item['score']:.4f}\n")

        with open(f"{base_name}_larger_clap.txt", "w", encoding="utf-8") as f:
            f.write("Larger-CLAP-General Results:\n")
            for item in larger_clap_result:
                f.write(f"{item['label']}: {item['score']:.4f}\n")

        print(f"Completed classification for {audio_file}")


if __name__ == "__main__":
    process_all_turns()
