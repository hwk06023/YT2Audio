import os
from textgrids import TextGrid
import torchaudio
import re


def split_words_timestamp(align_directory, text, waveform, sample_rate):
    tg = TextGrid()
    tg.read(f"{align_directory}/audio.TextGrid")
    word_items = tg["words"]

    words = []
    for i in range(len(word_items)):
        if word_items[i].text:
            words.append(word_items[i])

    extracted_words = re.findall(r"[가-힣\w]+", text)
    start_times = []
    end_times = []
    word_idx = 0
    print("text length: ", len(extracted_words))
    print("word length: ", len(words))

    for i, actual_word in enumerate(extracted_words):
        cur_idx = 0
        start_times.append(words[word_idx].xmin)
        while cur_idx < len(actual_word):
            cur_idx += len(words[word_idx].text)
            word_idx += 1
        end_times.append(words[word_idx - 1].xmax)

    output_dir = f"debug/time_stamp/{align_directory}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(extracted_words)):
        word_audio = waveform[
            :, int(start_times[i] * sample_rate) : int(end_times[i] * sample_rate)
        ]
        torchaudio.save(
            f"{output_dir}/{extracted_words[i]}.wav", word_audio, sample_rate
        )


audio_name = "mfa/짚톡 ) MBTI 토론 1회 (w.핑맨,김똘복,꽃핀)/"

for i in range(1, len(os.listdir(audio_name)) + 1):
    audio_path = audio_name + "processed_" + str(i) + "/audio.wav"
    text_path = audio_name + "processed_" + str(i) + "/audio.lab"
    align_directory = audio_name + "processed_" + str(i) + "_align"

    try:
        with open(text_path, "r") as f:
            text = f.read()
        waveform, sample_rate = torchaudio.load(audio_path)
        split_words_timestamp(align_directory, text, waveform, sample_rate)
    except FileNotFoundError as e:
        print(f"Skipping processed_{i}: {e}")
        continue
