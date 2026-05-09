import os
import json
import shutil
import time
import numpy as np
import librosa
import soundfile as sf
import requests

SOURCE_DIR = "cries/pokemon/latest"
DATA_RAW_DIR = "data_raw"
DATA_DIR = "data"
MODEL_DIR = "models"
CACHE_PATH = "models/pokemon_names.json"

N_MFCC = 13
MAX_LEN = 80
TARGET_SR = 16000

MIN_POKEMON_ID = 1
MAX_POKEMON_ID = 1025
AUGMENT_PER_POKEMON = 30

def load_name_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_name_cache(cache):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def sanitize_filename(name):
    invalid_chars = r'<>:"/\|?*'

    for char in invalid_chars:
        name = name.replace(char, "_")

    name = name.replace("♀", "_F")
    name = name.replace("♂", "_M")
    name = name.replace("'", "")
    name = name.strip()

    return name


def extract_multilingual_names(species_data):
    names = {
        "en": None,
        "zh": None,
        "ja": None
    }

    for item in species_data.get("names", []):
        lang = item["language"]["name"]
        name = item["name"]

        if lang == "en":
            names["en"] = name
        elif lang.startswith("zh") and names["zh"] is None:
            names["zh"] = name
        elif lang.startswith("ja") and names["ja"] is None:
            names["ja"] = name

    return names


def get_pokemon_names(pokemon_id, cache):
    key = str(pokemon_id)

    if key in cache:
        cached_value = cache[key]

        if isinstance(cached_value, str):
            names = {
                "en": cached_value,
                "zh": cached_value,
                "ja": cached_value
            }
            cache[key] = names
            save_name_cache(cache)
            return names

        return cached_value

    url = f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_id}"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            names = {
                "en": f"Pokemon_{pokemon_id}",
                "zh": f"Pokemon_{pokemon_id}",
                "ja": f"Pokemon_{pokemon_id}"
            }
        else:
            species_data = response.json()
            names = extract_multilingual_names(species_data)

            if names["en"] is None:
                names["en"] = f"Pokemon_{pokemon_id}"
            if names["zh"] is None:
                names["zh"] = names["en"]
            if names["ja"] is None:
                names["ja"] = names["en"]

        cache[key] = names
        save_name_cache(cache)
        time.sleep(0.1)

        return names

    except Exception as e:
        print(f"[API Error] ID {pokemon_id}: {e}")

        names = {
            "en": f"Pokemon_{pokemon_id}",
            "zh": f"Pokemon_{pokemon_id}",
            "ja": f"Pokemon_{pokemon_id}"
        }

        cache[key] = names
        save_name_cache(cache)
        return names


def make_folder_name(pokemon_id, english_name):
    safe_name = sanitize_filename(english_name)
    return f"{pokemon_id:03d}_{safe_name}"


def extract_id_from_folder(folder_name):
    id_part = folder_name.split("_", 1)[0]

    if id_part.isdigit():
        return str(int(id_part))

    return None


def get_label_from_folder(folder_name, name_cache):
    pokemon_id = extract_id_from_folder(folder_name)

    if pokemon_id and pokemon_id in name_cache:
        cached_value = name_cache[pokemon_id]

        if isinstance(cached_value, dict):
            return cached_value.get("en", folder_name)

        if isinstance(cached_value, str):
            return cached_value

    if "_" in folder_name:
        return folder_name.split("_", 1)[1]

    return folder_name


def get_all_pokemon_from_cries():
    cache = load_name_cache()
    pokemon_ids = []

    for file_name in os.listdir(SOURCE_DIR):
        if not file_name.endswith(".ogg"):
            continue

        pokemon_id = os.path.splitext(file_name)[0]

        if not pokemon_id.isdigit():
            continue

        pokemon_id = int(pokemon_id)

        if MIN_POKEMON_ID <= pokemon_id <= MAX_POKEMON_ID:
            pokemon_ids.append(pokemon_id)

    pokemon_ids.sort()

    pokemon_list = []

    for pokemon_id in pokemon_ids:
        names = get_pokemon_names(pokemon_id, cache)
        english_name = names["en"]
        folder_name = make_folder_name(pokemon_id, english_name)

        pokemon_list.append({
            "id": pokemon_id,
            "folder": folder_name,
            "en": english_name,
            "zh": names["zh"],
            "ja": names["ja"]
        })

    return pokemon_list


def prepare_raw_data(pokemon_list):
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    for pokemon in pokemon_list:
        pokemon_id = pokemon["id"]
        folder_name = pokemon["folder"]

        source_file = os.path.join(SOURCE_DIR, f"{pokemon_id}.ogg")
        target_dir = os.path.join(DATA_RAW_DIR, folder_name)
        target_file = os.path.join(target_dir, f"{pokemon_id}.ogg")

        if not os.path.exists(source_file):
            print(f"[Missing] {source_file}")
            continue

        os.makedirs(target_dir, exist_ok=True)

        if os.path.exists(target_file):
            print(f"[Skip raw] {folder_name} already exists.")
            continue

        shutil.copy(source_file, target_file)
        print(f"[Copied] {folder_name}: {pokemon_id}.ogg")


def augment_audio(y, sr):
    y_aug = y.copy()

    # Random volume
    volume_factor = np.random.uniform(0.6, 1.4)
    y_aug = y_aug * volume_factor

    # Random noise
    if np.random.rand() < 0.7:
        noise_level = np.random.uniform(0.001, 0.01)
        noise = np.random.randn(len(y_aug))
        y_aug = y_aug + noise_level * noise

    # Random speed
    if np.random.rand() < 0.6:
        rate = np.random.uniform(0.85, 1.15)
        y_aug = librosa.effects.time_stretch(y_aug, rate=rate)

    # Random pitch
    if np.random.rand() < 0.6:
        steps = np.random.uniform(-2.0, 2.0)
        y_aug = librosa.effects.pitch_shift(
            y_aug,
            sr=sr,
            n_steps=steps
        )

    # Time shift
    if np.random.rand() < 0.5:
        shift = np.random.randint(-int(0.1 * sr), int(0.1 * sr))
        y_aug = np.roll(y_aug, shift)

    # Mild clipping distortion
    if np.random.rand() < 0.3:
        y_aug = np.clip(y_aug, -0.8, 0.8)

    # Normalize
    max_amp = np.max(np.abs(y_aug))
    if max_amp > 0:
        y_aug = y_aug / max_amp

    return y_aug


def augment_one_pokemon(folder_name):
    raw_dir = os.path.join(DATA_RAW_DIR, folder_name)
    output_dir = os.path.join(DATA_DIR, folder_name)

    if not os.path.isdir(raw_dir):
        return

    os.makedirs(output_dir, exist_ok=True)

    existing_wavs = [
        f for f in os.listdir(output_dir)
        if f.endswith(".wav")
    ]

    if len(existing_wavs) >= AUGMENT_PER_POKEMON:
        print(f"[Skip augment] {folder_name} already has enough data.")
        return

    ogg_files = [
        f for f in os.listdir(raw_dir)
        if f.endswith(".ogg")
    ]

    if not ogg_files:
        print(f"[No audio] {folder_name}")
        return

    input_path = os.path.join(raw_dir, ogg_files[0])
    y, sr = librosa.load(input_path, sr=TARGET_SR)

    y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) == 0:
        print(f"[Too quiet] {folder_name}")
        return

    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp

    safe_name = folder_name.lower()

    original_path = os.path.join(output_dir, f"{safe_name}_000_original.wav")

    if not os.path.exists(original_path):
        sf.write(original_path, y, TARGET_SR)

    for i in range(1, AUGMENT_PER_POKEMON):
        output_path = os.path.join(
            output_dir,
            f"{safe_name}_{i:03d}_aug.wav"
        )

        if os.path.exists(output_path):
            continue

        y_aug = augment_audio(y, TARGET_SR)

        sf.write(output_path, y_aug, TARGET_SR)

    print(f"[Augmented] {folder_name}")


def build_dataset():
    features = []
    labels = []
    name_cache = load_name_cache()

    if not os.path.isdir(DATA_DIR):
        print("No data folder found.")
        return

    pokemon_folders = sorted([
        folder for folder in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, folder))
    ])

    for folder_name in pokemon_folders:
        label_path = os.path.join(DATA_DIR, folder_name)
        label_name = get_label_from_folder(folder_name, name_cache)

        for file_name in os.listdir(label_path):
            if not file_name.endswith(".wav"):
                continue

            file_path = os.path.join(label_path, file_name)

            y, sr = librosa.load(file_path, sr=TARGET_SR)

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=N_MFCC
            )

            if mfcc.shape[1] < MAX_LEN:
                pad_width = MAX_LEN - mfcc.shape[1]
                mfcc = np.pad(
                    mfcc,
                    ((0, 0), (0, pad_width)),
                    mode="constant"
                )
            else:
                mfcc = mfcc[:, :MAX_LEN]

            features.append(mfcc)
            labels.append(label_name)

    features = np.array(features)
    labels = np.array(labels)

    os.makedirs(MODEL_DIR, exist_ok=True)

    np.save(os.path.join(MODEL_DIR, "features.npy"), features)
    np.save(os.path.join(MODEL_DIR, "labels.npy"), labels)

    print("Dataset saved successfully!")
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
    print("Number of classes:", len(set(labels)))
    print("Classes:", sorted(set(labels)))


def main():
    print("Reading Pokémon cries...")
    pokemon_list = get_all_pokemon_from_cries()

    print(f"Found {len(pokemon_list)} Pokémon.")

    print("\nPreparing raw data...")
    prepare_raw_data(pokemon_list)

    print("\nGenerating augmented data...")
    for pokemon in pokemon_list:
        augment_one_pokemon(pokemon["folder"])

    print("\nBuilding dataset...")
    build_dataset()

    print("\nName cache saved to:")
    print(CACHE_PATH)


if __name__ == "__main__":
    main()