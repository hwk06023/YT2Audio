from preprocess.elevenlabs_split_turns import ElevenLabs
import os
import traceback
from dotenv import load_dotenv
from pydub import AudioSegment
import io
import argparse
import json
from itertools import groupby

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

directory_name = ""  # enter directory name

for i in range(len(os.listdir("data/" + directory_name))):
    file_path = "data/" + directory_name + "/processed_" + str(i + 1) + ".wav"

    if os.path.exists(file_path):
        audio = AudioSegment.from_wav(file_path)
    else:
        print(f"파일이 존재하지 않습니다: {file_path}")
        continue

    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)

    print("음성을 텍스트로 변환 중...")
    print("화자 분리(다이어라이즈) 옵션:켜짐")

    result = client.speech_to_text.convert(
        file=buffer,
        model_id="scribe_v1",
        tag_audio_events=True,
        diarize=True,
    )

    print("\n--- 변환 결과 ---")
    print("Transcription text:", result.text)

    # 디렉토리 확인 및 생성
    output_dir = os.path.dirname("wavdata/data/transcription_result.txt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"디렉토리가 생성되었습니다: {output_dir}")

    # 텍스트 저장
    output_file = "data/transcription_result.txt"

    text_to_save = ""
    if hasattr(result, "words") and result.words:
        formatted_text_parts = []
        current_speaker = None
        line_buffer = ""

        for word in result.words:
            if word.type == "word":
                speaker = word.speaker_id if hasattr(word, "speaker_id") else "Unknown"
                if speaker != current_speaker:
                    if line_buffer:  # Append previous line buffer
                        formatted_text_parts.append(line_buffer.strip())
                    line_buffer = (
                        f"[Speaker {speaker}]: {word.text}"  # Start new line buffer
                    )
                    current_speaker = speaker
                else:
                    if line_buffer:  # Add space only if buffer has content
                        line_buffer += f" {word.text}"
                    else:  # If buffer somehow empty, start it
                        line_buffer = f"[Speaker {speaker}]: {word.text}"

            elif word.type == "audio_event":
                if line_buffer:  # Append previous line buffer before event
                    formatted_text_parts.append(line_buffer.strip())
                    line_buffer = ""  # Reset buffer
                    current_speaker = None  # Reset speaker context for events
                formatted_text_parts.append(f"[{word.text}]")  # Add event
            # 'spacing' events are implicitly handled by adding spaces between words

        if line_buffer:  # Append any remaining buffer
            formatted_text_parts.append(line_buffer.strip())

        text_to_save = "\n".join(formatted_text_parts)
    else:
        # Fallback to raw text if not diarizing or no word data
        text_to_save = result.text

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text_to_save)

    # 저장 확인
    if os.path.exists(output_file):
        print(f"변환 결과가 {output_file}에 저장되었습니다.")
        print(f"파일 크기: {os.path.getsize(output_file)} 바이트")
    else:
        print(f"파일이 생성되지 않았습니다: {output_file}")

    if hasattr(result, "words") and result.words:
        print("\nWords with timing and speaker info:")
        speakers = set()
        audio_events = []

        for word in result.words:
            if word.type == "audio_event":
                print(f"- Audio event: {word.text} at {word.start}s to {word.end}s")
                audio_events.append(word)
            elif word.type == "word":
                print(
                    f"- Word: '{word.text}' (Speaker {word.speaker_id}) at {word.start}s to {word.end}s"
                )
                speakers.add(word.speaker_id)

        # 화자 수와 오디오 이벤트 숫자도 저장
        with open(
            "wavdata/data/transcription_metadata.txt", "w", encoding="utf-8"
        ) as f:
            f.write(f"Total speakers detected: {len(speakers)}\n")
            for speaker in speakers:
                f.write(f"- {speaker}\n")
            f.write(f"\nTotal audio events detected: {len(audio_events)}\n")
            for event in audio_events:
                f.write(f"- {event.text} at {event.start}s to {event.end}s\n")
        print("\n메타데이터가 저장되었습니다: wavdata/data/transcription_metadata.txt")

        print(f"\nTotal speakers detected: {len(speakers)}")
        for speaker in speakers:
            print(f"- {speaker}")

        # A-B-A 3턴 형태로 분리
        print("\nA-B-A 3턴 형태로 분리 중...")

        # 1. 연속된 동일 화자의 발화를 그룹화
        speaker_segments = []
        word_segments = []

        # 오직 word 타입만 처리
        word_only = [word for word in result.words if word.type == "word"]

        # 화자별로 그룹화
        for speaker_id, group in groupby(word_only, key=lambda x: x.speaker_id):
            group_list = list(group)
            combined_text = " ".join([word.text for word in group_list])
            start_time = group_list[0].start
            end_time = group_list[-1].end

            segment = {
                "Text": combined_text,
                "Starttime": start_time,
                "Endtime": end_time,
                "Speaker": speaker_id,
            }

            speaker_segments.append(segment)

        # 2. A-B-A 패턴 찾기
        turns_output_dir = (
            f"data/{directory_name}" + "/processed_" + str(i + 1) + "/turns"
        )
        if not os.path.exists(turns_output_dir):
            os.makedirs(turns_output_dir)

        turn_count = 0

        # 화자 순서가 달라지는 패턴 찾기 (A-B-A)
        i = 0
        while i < len(speaker_segments) - 2:
            speaker_a = speaker_segments[i]["Speaker"]
            speaker_b = speaker_segments[i + 1]["Speaker"]

            # A-B-A 패턴 확인
            if (
                speaker_a != speaker_b
                and speaker_segments[i + 2]["Speaker"] == speaker_a
            ):
                turn_data = [
                    speaker_segments[i],
                    speaker_segments[i + 1],
                    speaker_segments[i + 2],
                ]

                # 3턴 전체 시간 범위
                start_time_sec = turn_data[0]["Starttime"]
                end_time_sec = turn_data[2]["Endtime"]

                # 밀리초 단위로 변환 (오디오 자르기용)
                start_time_ms = int(start_time_sec * 1000)
                end_time_ms = int(end_time_sec * 1000)

                # 오프셋(60초) 고려하여 원본 오디오에서의 위치 계산
                original_start_ms = (
                    start_time_ms + start_time
                )  # start_time(60초)는 처음에 자른 시간
                original_end_ms = end_time_ms + start_time

                # JSON 파일 저장
                turn_count += 1
                json_filename = f"{turns_output_dir}/turn_{turn_count}.json"

                with open(json_filename, "w", encoding="utf-8") as f:
                    json.dump(turn_data, f, ensure_ascii=False, indent=4)

                # 오디오 세그먼트 자르기 및 저장
                audio_segment = audio[original_start_ms:original_end_ms]
                audio_filename = f"{turns_output_dir}/turn_{turn_count}.wav"
                audio_segment.export(audio_filename, format="wav")

                print(f"3턴 분할 {turn_count}: {json_filename}, {audio_filename}")
                print(f"  - 화자 패턴: {speaker_a}-{speaker_b}-{speaker_a}")
                print(f"  - 시간: {start_time_sec:.2f}s - {end_time_sec:.2f}s")

                # 다음 검색 위치로 이동 (이미 사용한 턴 건너뛰기)
                i += 3
            else:
                # 패턴이 맞지 않으면 다음 위치로
                i += 1

        print(f"\n총 {turn_count}개의 A-B-A 3턴 세그먼트를 생성했습니다.")
