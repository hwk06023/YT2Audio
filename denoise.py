# Deprecated

from denoiser import pretrained
from denoiser.dsp import convert_audio
import os
import torch
import torchaudio
import gc

denoiser = pretrained.dns64()
if torch.cuda.is_available():
    denoiser.cuda()

audio_name = ""
for i in range(1, len(os.listdir(audio_name)) + 1):
    audio_path = audio_name + "processed_" + str(i) + ".wav"
    denoise_path = audio_name + "processed_" + str(i) + "_denoised.wav"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Denoise audio -------------------------------------------------------
    if os.path.exists(denoise_path):
        denoised_stereo, _ = torchaudio.load(denoise_path)
        print("Denoised audio detected. Skipping denoising...")
    else:
        print("Denoising audio...")
        waveform, sr = torchaudio.load(audio_path)
        waveform = convert_audio(waveform, sr, denoiser.sample_rate, channels=2)
        waveform = waveform.to(device)

        denoised_channels = []
        with torch.no_grad():
            for i in [0, 1]:
                denoised = denoiser(waveform[i].unsqueeze(0).unsqueeze(0))[0, 0]
                denoised_channels.append(denoised.cpu())
                collected = gc.collect()

        denoised_stereo = torch.stack(denoised_channels, dim=0)
        os.makedirs(os.path.dirname(denoise_path), exist_ok=True)
        torchaudio.save(
            denoise_path, src=denoised_stereo, sample_rate=denoiser.sample_rate
        )
