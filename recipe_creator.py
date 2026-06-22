# Author: Mithil
import json
import ollama
import time

INPUT_FILE = "south_asian_corpus_enriched.json"  # Your current file
OUTPUT_FILE = "vector_ready_corpus.json"
MODEL_NAME = "llama3"

def extract_recipe_json(raw_text: str) -> dict:
    """
    Forces Llama 3 to read the raw text and extract it into a strict JSON object.
    """
    system_prompt = """You are a strict Data Extraction API.
Read the provided recipe text and extract the introduction, ingredients, and instructions.
You MUST output ONLY a valid JSON object matching this exact schema:
{
    "intro": "A brief string summarizing the recipe background.",
    "ingredients": ["item 1", "item 2", "item 3"],
    "instructions": ["step 1", "step 2", "step 3"]
}
Do not include markdown blocks, just the raw JSON."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_text}
    ]

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            format='json',
            options={"temperature": 0.0}
        )

        return json.loads(response['message']['content'])
    except Exception as e:
        print(f"  [ERROR] Failed to extract JSON: {e}")
        return {"intro": "", "ingredients": [], "instructions": []}

def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {INPUT_FILE}.")
        return

    formatted_data = []
    total_recipes = len(data)

    print(f"Found {total_recipes} recipes. Starting JSON extraction via Ollama...\n")

    for index, item in enumerate(data, 1):
        title = item.get("title", "Unknown Dish")
        print(f"[{index}/{total_recipes}] Processing: {title}...")

        raw_text = item.get("full_text", "")
        
        recipe_dict = extract_recipe_json(raw_text)

        structured_item = {
            "id": item.get("id", f"dish_{index}"),
            "source_url": item.get("source_url", ""),
            "title": title,
            "cuisine_type": item.get("cuisine_type", "South Asian"),
            "full_text": raw_text,
            "recipe": recipe_dict,
            "metadata": item.get("metadata", {})
        }

        formatted_data.append(structured_item)
        time.sleep(0.5)

    print(f"\nSaving structured JSON data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)

    print("✅ Process complete!")

if __name__ == "__main__":
    main()