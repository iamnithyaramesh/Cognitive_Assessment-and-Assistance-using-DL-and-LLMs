import requests
import os
import time
import re
from dotenv import load_dotenv
from fuzzywuzzy import fuzz

load_dotenv()
sarvam_api_key = os.getenv("SARVAM_API_KEY")
API_URL = "https://api.sarvam.ai/v1/chat/completions"
list_address=[]

def calculate_score(threshold=85):
    print("\n--- Now it's time to recall the addresses ---")
    score = 0

    for i, actual_address in enumerate(list_address):
        user_input = input(f"\nRecall address #{i + 1}: ").strip()

        similarity = fuzz.ratio(user_input.lower(), actual_address.lower())
        print(f"Similarity: {similarity}%")

        if similarity >= threshold:
            print("Close enough! Point awarded.")
            score += 1
        else:
            print(f"Not close enough.\nExpected: {actual_address}")
    return score


def main():
    headers = {
        "api-subscription-key": sarvam_api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sarvam-m",
        "messages": [
            {"role": "user", "content": "Generate only three realistic Indian residential address in India in one paragraph. "
                                        "Do not include explanations or multiple formats. "
                                        "No markdown formatting or additional notes."}
        ]
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        raw_output = result["choices"][0]["message"]["content"]

        addresses = re.findall(r"\d+\.\s+(.*)", raw_output)

        print("\nRemember these generated addresses:\n")
        for i, address in enumerate(addresses):
            clean_address = address.replace("*", "").strip()
            list_address.append(clean_address)
            print(f"{i + 1}. {clean_address}")
            time.sleep(20)
    else:
        print("Error:", response.status_code, response.text)
        return

    score = calculate_score()
    print(f"\nYour final score: {score}/{len(list_address)}")

if __name__ == "__main__":
    main()
