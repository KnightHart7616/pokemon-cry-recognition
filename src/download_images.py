import os
import json
import time
import requests

NAMES_PATH = "models/pokemon_names.json"
IMAGE_DIR = "images"

BASE_URL = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
    "sprites/pokemon/other/official-artwork"
)


def load_pokemon_names():
    if not os.path.exists(NAMES_PATH):
        raise FileNotFoundError(f"Cannot find {NAMES_PATH}")

    with open(NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def download_image(pokemon_id, pokemon_name):
    os.makedirs(IMAGE_DIR, exist_ok=True)

    image_path = os.path.join(IMAGE_DIR, f"{pokemon_id}.png")

    if os.path.exists(image_path):
        print(f"[Skip] {pokemon_id} {pokemon_name} already exists.")
        return

    url = f"{BASE_URL}/{pokemon_id}.png"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(response.content)

            print(f"[Downloaded] {pokemon_id} {pokemon_name}")

        else:
            print(f"[Failed] {pokemon_id} {pokemon_name}: HTTP {response.status_code}")

    except Exception as e:
        print(f"[Error] {pokemon_id} {pokemon_name}: {e}")


def main():
    pokemon_names = load_pokemon_names()

    print(f"Found {len(pokemon_names)} Pokémon in cache.")

    for pokemon_id, names in sorted(
        pokemon_names.items(),
        key=lambda item: int(item[0])
    ):
        pokemon_name = names.get("en", f"Pokemon_{pokemon_id}")
        download_image(pokemon_id, pokemon_name)
        time.sleep(0.05)

    print("\nImage download completed.")


if __name__ == "__main__":
    main()