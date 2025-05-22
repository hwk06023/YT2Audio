import os
from pydub import AudioSegment

file_name = ""  # enter file name
file_path = "data/" + file_name + ".wav"

if os.path.exists(file_path):
    audio = AudioSegment.from_wav(file_path)

    start_time = 60 * 1000
    end_time = len(audio) - 60 * 1000
    segment = audio[start_time:end_time]

    segment_length = 30 * 60 * 1000
    num_segments = (end_time - start_time) // segment_length + 1

    output_dir = f"data/{file_name}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_segments):
        segment_start = start_time + (i * segment_length)
        segment_end = min(segment_start + segment_length, end_time)
        current_segment = audio[segment_start:segment_end]

        temp_file_path = f"{output_dir}/processed_{i+1}.wav"
        current_segment.export(temp_file_path, format="wav")

        print(
            f"구간 {i+1}/{num_segments}이 {temp_file_path}로 저장되었습니다. (시작: {segment_start/1000}초, 종료: {segment_end/1000}초)"
        )
