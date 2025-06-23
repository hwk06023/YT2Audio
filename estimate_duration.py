import subprocess
import json
import re


def get_duration_from_url(url):
    try:
        command = [
            "yt-dlp",
            url,
            "--dump-json",
            "--no-download",
            "--cookies",
            "cookies_1.txt",
            "--flat-playlist",
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        total_seconds = 0
        video_count = 0

        for line in result.stdout.strip().split("\n"):
            if line:
                try:
                    data = json.loads(line)
                    if "duration" in data and data["duration"]:
                        total_seconds += data["duration"]
                        video_count += 1
                except json.JSONDecodeError:
                    continue

        return total_seconds, video_count
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return 0, 0


def main():
    with open("lists/korean.txt", "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    total_seconds = 0
    total_videos = 0

    for url in urls:
        print(f"Processing: {url}")
        seconds, count = get_duration_from_url(url)
        total_seconds += seconds
        total_videos += count

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    result = (
        f"Total: {total_videos} videos, {hours}h {minutes}m ({total_seconds} seconds)"
    )

    with open("duration_result.txt", "w") as f:
        f.write(result)

    print(result)


if __name__ == "__main__":
    main()
