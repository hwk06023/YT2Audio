import os
import subprocess
import re
import json

audio_format = "wav"
CONTAINER_SIZE = 50

lang_files = {
    "english": "lists/english.txt",
    "korean": "lists/korean.txt",
    # "japanese": "lists/japanese.txt",
    # "chinese": "lists/chinese.txt",
}


def sanitize_folder_name(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)


def get_video_duration(url):
    command = [
        "yt-dlp",
        url,
        "--dump-json",
        "--no-download",
        "--cookies",
        "cookies.txt",
    ]    

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    video_info = json.loads(result.stdout)
    return video_info['duration']


def create_sections_str(duration):
    sections = []
    for start in range(0, duration, 1800):
        end = min(start + 1800, duration)
        start_time = f"{start//3600}:{(start%3600)//60:02d}:{start%60:02d}"
        end_time = f"{end//3600}:{(end%3600)//60:02d}:{end%60:02d}"
        sections.append(f"*{start_time}-{end_time}")
    
    return ",".join(sections)


def load_endpoint(lang):
    endpoint_file = f"/mnt/data/{lang}/wav/endpoint.json"
    if os.path.exists(endpoint_file):
        with open(endpoint_file, 'r') as f:
            return json.load(f)
    return {"url_index": 0, "container_index": 1, "container_count": 0}


def save_endpoint(lang, url_index, container_index, container_count):
    endpoint_file = f"/mnt/data/{lang}/wav/endpoint.json"
    with open(endpoint_file, 'w') as f:
        json.dump({"url_index": url_index, "container_index": container_index, "container_count": container_count}, f)


def update_full_metadata(lang, container_index, url, videos):
    metadata_file = f"/mnt/data/{lang}/wav/full_metadata.txt"
    with open(metadata_file, 'a') as f:
        f.write(f"Container {container_index}: URL {url}\n")
        for video in videos:
            f.write(f"  - {video}\n")
        f.write("\n")


def download_container(lang, urls, start_url_idx, container_index):
    container_dir = f"/mnt/data/{lang}/wav/container_{container_index}"
    os.makedirs(container_dir, exist_ok=True)
    
    downloaded_count = 0
    current_url_idx = start_url_idx
    
    while downloaded_count < CONTAINER_SIZE and current_url_idx < len(urls):
        url = urls[current_url_idx]
        print(f"Processing URL {current_url_idx + 1}: {url}")
        
        duration = get_video_duration(url)
        sections_str = create_sections_str(duration)
        
        command = [
            "yt-dlp", url, "-x", "--audio-format", audio_format, "--audio-quality", "0",
            "--output", os.path.join(container_dir, "%(title)s.%(ext)s"),
            "--ignore-errors", "--no-overwrites", "--cookies", "cookies.txt",
            "--download-sections", sections_str, "--max-downloads", str(CONTAINER_SIZE - downloaded_count)
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        downloaded_files = [f for f in os.listdir(container_dir) if f.endswith(f'.{audio_format}')]
        current_downloaded = len(downloaded_files)
        
        update_full_metadata(lang, container_index, url, downloaded_files[downloaded_count:current_downloaded])
        
        with open(f"{container_dir}/container_metadata.txt", 'w') as f:
            f.write(f"Container {container_index} - Progress: {current_downloaded}/{CONTAINER_SIZE}\n")
            for file in downloaded_files:
                f.write(f"✓ {file}\n")
        
        downloaded_count = current_downloaded
        
        if downloaded_count < CONTAINER_SIZE:
            current_url_idx += 1
        else:
            break
    
    return current_url_idx, downloaded_count


def main():
    for lang, txt_path in lang_files.items():
        os.makedirs(f"/mnt/data/{lang}/wav", exist_ok=True)
        
        with open(txt_path, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
        
        endpoint = load_endpoint(lang)
        url_index = endpoint["url_index"]
        container_index = endpoint["container_index"]
        container_count = endpoint["container_count"]
        
        print(f"Resuming {lang}: URL {url_index + 1}, Container {container_index}")
        
        while url_index < len(urls):
            if container_count < CONTAINER_SIZE:
                next_url_idx, downloaded = download_container(lang, urls, url_index, container_index)
                
                if downloaded >= CONTAINER_SIZE:
                    container_index += 1
                    container_count = 0
                    url_index = next_url_idx
                else:
                    url_index = next_url_idx
                    container_count = downloaded
            else:
                container_index += 1
                container_count = 0
            
            save_endpoint(lang, url_index, container_index, container_count)


if __name__ == "__main__":
    main()
