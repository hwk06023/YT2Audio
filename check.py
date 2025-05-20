import os
import wave
import contextlib

def get_total_wav_duration(directory='downloads'):
    total_duration = 0
    wav_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    for wav_file in wav_files:
        try:
            with contextlib.closing(wave.open(wav_file, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                total_duration += duration
                print(f"{wav_file}: {duration:.2f} seconds")
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
    
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = total_duration % 60
    
    print(f"\nTotal WAV files found: {len(wav_files)}")
    print(f"Total duration: {hours}h {minutes}m {seconds:.2f}s ({total_duration:.2f} seconds)")

if __name__ == "__main__":
    get_total_wav_duration()
