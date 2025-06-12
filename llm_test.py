import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def classify_with_llm(tag_name):
    prompt = f"""Classify the following audio tag as either paralinguistic or non-paralinguistic sound.
Definitions:
- Paralinguistic: Non-verbal human sounds (laughter, coughs, sighs, interjections)
- Non-paralinguistic: Speech, music, sound effects, mechanical/environmental sounds

Tag to classify: {tag_name}

Response: Either 'paralinguistic' or 'non_paralinguistic'"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,
    )
    return response.choices[0].message.content.strip().lower()


PARALINGUISTIC_EVENTS = [
    "laughter",
    "chuckle",
    "giggle",
    "snicker",
    "crying",
    "gasp",
    "sigh",
    "cough",
    "sneeze",
    "throat_clearing",
    "breathing",
    "yawn",
    "snoring",
    "grunt",
    "groan",
    "hiccup",
    "burping",
    "hum",
    "humming",
    "whistle",
    "whistling",
    "whispering",
    "pant",
    "sniff",
    "babbling",
    "chewing",
    "screaming",
    "shout",
    "yell",
]

NON_PARALINGUISTIC_EVENTS = [
    "music",
    "applause",
    "door_slam",
    "speech",
    "footsteps",
    "rain",
    "thunder",
    "bell",
    "phone_ring",
    "car_engine",
    "typing",
    "clock_ticking",
    "alarm",
    "siren",
    "dog_bark",
    "bird_chirp",
    "water_drip",
    "wind",
    "fire_crackling",
    "glass_break",
    "paper_rustle",
    "zipper",
    "keyboard",
    "mouse_click",
]

ground_truth = {}
for tag in PARALINGUISTIC_EVENTS:
    ground_truth[tag] = "paralinguistic"
for tag in NON_PARALINGUISTIC_EVENTS:
    ground_truth[tag] = "non_paralinguistic"

test_tags = PARALINGUISTIC_EVENTS + NON_PARALINGUISTIC_EVENTS

correct = 0
total = len(test_tags)

for tag in test_tags:
    result = classify_with_llm(tag)
    expected = ground_truth[tag]
    is_correct = result == expected
    correct += is_correct
    status = "✓" if is_correct else "✗"
    print(f"{status} '{tag}' -> {result} (expected: {expected})")

accuracy = correct / total * 100
print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
