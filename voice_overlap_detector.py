import numpy as np
import librosa
import scipy.signal
from typing import List, Tuple, Optional
import webrtcvad
import wave


class VoiceOverlapDetector:
    def __init__(self, sample_rate: int = 16000, frame_duration: int = 30):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.vad = webrtcvad.Vad(3)

    def detect_voice_activity(
        self, audio_data: np.ndarray
    ) -> List[Tuple[float, float]]:
        frame_size = int(self.sample_rate * self.frame_duration / 1000)
        voice_segments = []
        current_start = None

        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame = audio_data[i : i + frame_size]
            audio_bytes = (frame * 32767).astype(np.int16).tobytes()

            if self.vad.is_speech(audio_bytes, self.sample_rate):
                if current_start is None:
                    current_start = i / self.sample_rate
            else:
                if current_start is not None:
                    voice_segments.append((current_start, i / self.sample_rate))
                    current_start = None

        if current_start is not None:
            voice_segments.append((current_start, len(audio_data) / self.sample_rate))

        return voice_segments

    def energy_based_detection(
        self,
        audio_data: np.ndarray,
        window_size: float = 0.025,
        hop_size: float = 0.010,
        threshold_percentile: int = 30,
    ) -> List[Tuple[float, float]]:
        win_length = int(window_size * self.sample_rate)
        hop_length = int(hop_size * self.sample_rate)

        energy = []
        for i in range(0, len(audio_data) - win_length, hop_length):
            frame = audio_data[i : i + win_length]
            frame_energy = np.sum(frame**2)
            energy.append(frame_energy)

        energy = np.array(energy)
        threshold = np.percentile(energy, threshold_percentile)

        voice_frames = energy > threshold
        segments = []
        in_voice = False
        start_time = 0

        for i, is_voice in enumerate(voice_frames):
            time = i * hop_size
            if is_voice and not in_voice:
                start_time = time
                in_voice = True
            elif not is_voice and in_voice:
                segments.append((start_time, time))
                in_voice = False

        if in_voice:
            segments.append((start_time, len(voice_frames) * hop_size))

        return segments

    def spectral_centroid_detection(
        self, audio_data: np.ndarray
    ) -> List[Tuple[float, float]]:
        stft = librosa.stft(audio_data, sr=self.sample_rate)
        spectral_centroids = librosa.feature.spectral_centroid(
            S=np.abs(stft), sr=self.sample_rate
        )[0]

        threshold = np.mean(spectral_centroids) + np.std(spectral_centroids)
        voice_frames = spectral_centroids > threshold

        segments = []
        in_voice = False
        start_idx = 0

        hop_length = 512
        for i, is_voice in enumerate(voice_frames):
            time = librosa.frames_to_time(i, sr=self.sample_rate, hop_length=hop_length)
            if is_voice and not in_voice:
                start_time = time
                in_voice = True
            elif not is_voice and in_voice:
                segments.append((start_time, time))
                in_voice = False

        if in_voice:
            end_time = librosa.frames_to_time(
                len(voice_frames), sr=self.sample_rate, hop_length=hop_length
            )
            segments.append((start_time, end_time))

        return segments

    def detect_overlaps(
        self, segments1: List[Tuple[float, float]], segments2: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        overlaps = []
        for start1, end1 in segments1:
            for start2, end2 in segments2:
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                if overlap_start < overlap_end:
                    overlaps.append((overlap_start, overlap_end))
        return overlaps

    def process_stereo_audio(self, audio_file: str) -> dict:
        audio_data, sr = librosa.load(audio_file, sr=self.sample_rate, mono=False)

        if len(audio_data.shape) == 1:
            raise ValueError("Audio file must be stereo")

        left_channel = audio_data[0]
        right_channel = audio_data[1]

        left_segments = self.detect_voice_activity(left_channel)
        right_segments = self.detect_voice_activity(right_channel)

        overlaps = self.detect_overlaps(left_segments, right_segments)

        return {
            "left_channel_voice": left_segments,
            "right_channel_voice": right_segments,
            "overlaps": overlaps,
            "overlap_duration": sum(end - start for start, end in overlaps),
            "total_duration": len(audio_data[0]) / self.sample_rate,
        }

    def advanced_overlap_detection(self, audio_file: str, method: str = "vad") -> dict:
        audio_data, sr = librosa.load(audio_file, sr=self.sample_rate, mono=False)

        if len(audio_data.shape) == 1:
            audio_data = np.array([audio_data, audio_data])

        left_channel = audio_data[0]
        right_channel = audio_data[1]

        if method == "vad":
            left_segments = self.detect_voice_activity(left_channel)
            right_segments = self.detect_voice_activity(right_channel)
        elif method == "energy":
            left_segments = self.energy_based_detection(left_channel)
            right_segments = self.energy_based_detection(right_channel)
        elif method == "spectral":
            left_segments = self.spectral_centroid_detection(left_channel)
            right_segments = self.spectral_centroid_detection(right_channel)
        else:
            raise ValueError("Method must be 'vad', 'energy', or 'spectral'")

        overlaps = self.detect_overlaps(left_segments, right_segments)

        overlap_ratio = sum(end - start for start, end in overlaps) / (
            len(audio_data[0]) / self.sample_rate
        )

        return {
            "method": method,
            "left_voice_segments": left_segments,
            "right_voice_segments": right_segments,
            "overlap_segments": overlaps,
            "overlap_duration": sum(end - start for start, end in overlaps),
            "overlap_ratio": overlap_ratio,
            "total_duration": len(audio_data[0]) / self.sample_rate,
        }


def main():
    detector = VoiceOverlapDetector()

    audio_file = "sample_audio.wav"

    try:
        results = detector.advanced_overlap_detection(audio_file, method="vad")

        print(f"Detection Method: {results['method']}")
        print(f"Total Duration: {results['total_duration']:.2f} seconds")
        print(f"Overlap Duration: {results['overlap_duration']:.2f} seconds")
        print(f"Overlap Ratio: {results['overlap_ratio']:.2%}")
        print(f"\nOverlap Segments:")
        for i, (start, end) in enumerate(results["overlap_segments"]):
            print(f"  {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")

    except FileNotFoundError:
        print("Audio file not found. Please provide a valid stereo audio file.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
