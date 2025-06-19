import os
import subprocess
import re
import json
import math

audio_format = "wav"
CONTAINER_SIZE = 50
SEGMENT_DURATION = 1800

lang_files = {
    # "english": "lists/english.txt",
    "korean": "lists/test_korean.txt",
    # "japanese": "lists/japanese.txt",
    # "chinese": "lists/chinese.txt",
}


def sanitize_folder_name(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)


def get_playlist_videos(url):
    command = [
        "yt-dlp",
        url,
        "--dump-json",
        "--no-download",
        "--cookies",
        "cookies_1.txt",
        "--flat-playlist",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    videos = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            video_info = json.loads(line)
            if video_info.get("_type") != "playlist":
                videos.append(
                    {
                        "id": video_info.get("id"),
                        "title": video_info.get("title", "Unknown"),
                        "url": f"https://www.youtube.com/watch?v={video_info.get('id')}",
                        "duration": video_info.get("duration", 0),
                    }
                )
    print(f"get_playlist_videos: {videos}")
    return videos


def create_segments(videos):
    segments = []
    for video in videos:
        duration = video["duration"]
        num_segments = math.ceil(duration / SEGMENT_DURATION)
        for i in range(num_segments):
            start = i * SEGMENT_DURATION
            end = min(start + SEGMENT_DURATION, duration)
            start_time = f"{start//3600}:{(start%3600)//60:02d}:{start%60:02d}"
            end_time = f"{end//3600}:{(end%3600)//60:02d}:{end%60:02d}"

            segments.append(
                {
                    "video_id": video["id"],
                    "video_title": video["title"],
                    "video_url": video["url"],
                    "segment_index": i,
                    "start_time": start_time,
                    "end_time": end_time,
                    "section_str": f"*{start_time}-{end_time}",
                    "filename": f"{video['title']}_part{i+1:02d}.{audio_format}",
                }
            )
    return segments


def group_into_containers(segments):
    containers = []
    for i in range(0, len(segments), CONTAINER_SIZE):
        container_segments = segments[i : i + CONTAINER_SIZE]
        containers.append(
            {"index": i // CONTAINER_SIZE + 1, "segments": container_segments}
        )
    return containers


def save_metadata(lang, containers, all_videos):
    base_dir = f"../../{lang}/wav"
    os.makedirs(base_dir, exist_ok=True)

    with open(f"{base_dir}/full_metadata.txt", "w") as f:
        f.write(f"=== {lang.upper()} DATASET METADATA ===\n\n")
        f.write(f"Total Videos: {len(all_videos)}\n")
        f.write(f"Total Segments: {sum(len(c['segments']) for c in containers)}\n")
        f.write(f"Total Containers: {len(containers)}\n\n")

        for container in containers:
            f.write(
                f"Container {container['index']} ({len(container['segments'])} segments):\n"
            )
            for segment in container["segments"]:
                f.write(
                    f"  - {segment['video_title']} [{segment['start_time']}-{segment['end_time']}]\n"
                )
            f.write("\n")

    for container in containers:
        container_dir = f"{base_dir}/container_{container['index']}"
        os.makedirs(container_dir, exist_ok=True)

        with open(f"{container_dir}/container_metadata.txt", "w") as f:
            f.write(
                f"Container {container['index']} - 0/{len(container['segments'])} downloaded\n\n"
            )
            for segment in container["segments"]:
                f.write(f"⏳ {segment['filename']}\n")


def download_video_segments(video, video_segments, output_dir):
    video_id = video["id"]
    video_url = video["url"]

    segment_files = [
        os.path.join(output_dir, seg["filename"]) for seg in video_segments
    ]
    existing_files = [f for f in segment_files if os.path.exists(f)]

    if len(existing_files) == len(segment_files):
        return len(segment_files)

    ytdlp_cmd = [
        "yt-dlp",
        video_url,
        "--format",
        "bestaudio",
        "--quiet",
        "--no-warnings",
        "--cookies",
        "cookies_2.txt",
        "-o",
        "-",
    ]

    ffmpeg_outputs = []
    for segment in video_segments:
        output_path = os.path.join(output_dir, segment["filename"])
        if not os.path.exists(output_path):
            start_seconds = segment["segment_index"] * SEGMENT_DURATION
            ffmpeg_outputs.extend(
                [
                    "-ss",
                    str(start_seconds),
                    "-t",
                    str(SEGMENT_DURATION),
                    "-acodec",
                    "pcm_s16le" if audio_format == "wav" else "copy",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    output_path,
                ]
            )

    if not ffmpeg_outputs:
        return len(existing_files)

    ffmpeg_cmd = ["ffmpeg", "-y", "-i", "pipe:0", "-vn"] + ffmpeg_outputs

    ytdlp_proc = subprocess.Popen(
        ytdlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=ytdlp_proc.stdout, stderr=subprocess.DEVNULL
    )

    ytdlp_proc.stdout.close()

    ytdlp_proc.wait()
    ffmpeg_proc.wait()

    success_count = len([f for f in segment_files if os.path.exists(f)])
    return success_count


def update_container_metadata(lang, container, downloaded_count):
    container_dir = f"../../{lang}/wav/container_{container['index']}"
    with open(f"{container_dir}/container_metadata.txt", "w") as f:
        f.write(
            f"Container {container['index']} - {downloaded_count}/{len(container['segments'])} downloaded\n\n"
        )

        downloaded_files = set(os.listdir(container_dir))
        for segment in container["segments"]:
            status = "✓" if segment["filename"] in downloaded_files else "⏳"
            f.write(f"{status} {segment['filename']}\n")


def load_endpoint(lang):
    endpoint_file = f"../../{lang}/wav/endpoint.json"
    if os.path.exists(endpoint_file):
        with open(endpoint_file, "r") as f:
            return json.load(f)
    return {"container_index": 1, "segment_index": 0}


def save_endpoint(lang, container_index, segment_index):
    endpoint_file = f"../../{lang}/wav/endpoint.json"
    with open(endpoint_file, "w") as f:
        json.dump(
            {"container_index": container_index, "segment_index": segment_index}, f
        )


def main():
    for lang, txt_path in lang_files.items():
        print(f"\n=== Processing {lang.upper()} ===")
        os.makedirs(f"../../{lang}/wav", exist_ok=True)

        with open(txt_path, "r") as f:
            urls = [line.strip() for line in f if line.strip()]

        print("1. Collecting video metadata...")
        all_videos = []
        for url in urls:
            print(f"   Processing playlist: {url}")
            videos = get_playlist_videos(url)
            all_videos.extend(videos)

        print(f"   Found {len(all_videos)} videos total")

        print("2. Creating segments...")
        segments = create_segments(all_videos)
        print(f"   Created {len(segments)} segments")

        print("3. Grouping into containers...")
        containers = group_into_containers(segments)
        print(f"   Created {len(containers)} containers")

        print("4. Saving metadata...")
        save_metadata(lang, containers, all_videos)

        print("5. Starting downloads...")
        endpoint = load_endpoint(lang)
        start_container = endpoint["container_index"]
        start_segment = endpoint["segment_index"]

        for container in containers[start_container - 1 :]:
            container_dir = f"../../{lang}/wav/container_{container['index']}"
            print(f"   Downloading Container {container['index']}/{len(containers)}...")

            downloaded_count = 0

            # Group segments by video for efficient downloading
            video_segments_map = {}
            for segment in container["segments"]:
                video_id = segment["video_id"]
                if video_id not in video_segments_map:
                    video_segments_map[video_id] = []
                video_segments_map[video_id].append(segment)

            for video_id, video_segments in video_segments_map.items():
                video = next(v for v in all_videos if v["id"] == video_id)
                total_parts = len(video_segments)

                print(f"     Downloading {video['title']} ({total_parts} parts)...")

                success_count = download_video_segments(
                    video, video_segments, container_dir
                )
                downloaded_count += success_count

                update_container_metadata(lang, container, downloaded_count)
                save_endpoint(lang, container["index"], len(container["segments"]))

            start_segment = 0

        print(f"✓ Completed {lang}")


if __name__ == "__main__":
    main()
