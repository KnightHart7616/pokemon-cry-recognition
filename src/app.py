import os
import json
import joblib
import requests
import numpy as np
import librosa
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import sounddevice as sd
import soundfile as sf


MODEL_PATH = "models/pokemon_cry_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
NAMES_PATH = "models/pokemon_names.json"

N_MFCC = 13
MAX_LEN = 80
TARGET_SR = 16000

CONFIDENCE_THRESHOLD = 50.0

RECORDING_DIR = "recordings"
RECORDING_PATH = "recordings/latest_recording.wav"
RECORD_SECONDS = 3


def load_name_map():
    if not os.path.exists(NAMES_PATH):
        return {}

    with open(NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_feature(audio_path):
    y, sr = librosa.load(audio_path, sr=TARGET_SR)

    # Trim silence / background quiet parts
    y, _ = librosa.effects.trim(y, top_db=25)

    # If audio is too short after trimming, keep original behavior safe
    if len(y) == 0:
        raise ValueError("Audio is too quiet or empty.")

    # Normalize volume
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc.reshape(1, -1)


def normalize_name(label):
    if "_" in label:
        return label.split("_", 1)[1]
    return label


def find_multilingual_names(label, name_map):
    target_name = normalize_name(label).lower()

    for pokemon_id, names in name_map.items():
        en_name = names.get("en", "")

        if en_name.lower() == target_name:
            return {
                "id": pokemon_id,
                "en": names.get("en", label),
                "ja": names.get("ja", label),
                "zh": names.get("zh", label),
            }

    return {
        "id": "-",
        "en": label,
        "ja": label,
        "zh": label,
    }


def get_pokemon_image_path(pokemon_id):
    os.makedirs("images", exist_ok=True)
    return os.path.join("images", f"{pokemon_id}.png")


def download_pokemon_image(pokemon_id):
    if pokemon_id == "-":
        return None

    image_path = get_pokemon_image_path(pokemon_id)

    if os.path.exists(image_path):
        return image_path

    url = (
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        f"sprites/pokemon/other/official-artwork/{pokemon_id}.png"
    )

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(response.content)
            return image_path

        return None

    except Exception as e:
        print(f"Image download error: {e}")
        return None


class PokemonCryApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Pokémon Cry Recognition")
        self.geometry("720x620")
        self.resizable(False, False)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.model = joblib.load(MODEL_PATH)
        self.encoder = joblib.load(ENCODER_PATH)
        self.name_map = load_name_map()

        self.selected_audio_path = None

        self.build_ui()

    def build_ui(self):
        title = ctk.CTkLabel(
            self,
            text="Pokémon Cry Recognition",
            font=("Arial", 28, "bold")
        )
        title.pack(pady=(24, 4))

        subtitle = ctk.CTkLabel(
            self,
            text="Upload or record a Pokémon cry audio file and predict its species.",
            font=("Arial", 14)
        )
        subtitle.pack(pady=(0, 18))

        main_frame = ctk.CTkFrame(self, width=660, height=490)
        main_frame.pack(pady=10)
        main_frame.pack_propagate(False)

        left_frame = ctk.CTkFrame(main_frame, width=285, height=450)
        left_frame.pack(side="left", padx=(18, 9), pady=18)
        left_frame.pack_propagate(False)

        right_frame = ctk.CTkFrame(main_frame, width=330, height=450)
        right_frame.pack(side="right", padx=(9, 18), pady=18)
        right_frame.pack_propagate(False)

        left_title = ctk.CTkLabel(
            left_frame,
            text="Audio Input",
            font=("Arial", 20, "bold")
        )
        left_title.pack(pady=(24, 16))

        self.file_label = ctk.CTkLabel(
            left_frame,
            text="No audio file selected",
            font=("Arial", 13),
            wraplength=235
        )
        self.file_label.pack(pady=(10, 24))

        select_button = ctk.CTkButton(
            left_frame,
            text="Select Audio File",
            width=200,
            height=40,
            command=self.select_audio_file
        )
        select_button.pack(pady=8)

        record_button = ctk.CTkButton(
            left_frame,
            text=f"Record {RECORD_SECONDS} Seconds",
            width=200,
            height=40,
            command=self.record_audio
        )
        record_button.pack(pady=8)

        play_button = ctk.CTkButton(
            left_frame,
            text="Play Audio",
            width=200,
            height=40,
            command=self.play_audio
        )
        play_button.pack(pady=8)

        predict_button = ctk.CTkButton(
            left_frame,
            text="Predict",
            width=200,
            height=40,
            command=self.predict_audio
        )
        predict_button.pack(pady=8)

        hint = ctk.CTkLabel(
            left_frame,
            text="Supported formats:\n.wav / .ogg / .mp3",
            font=("Arial", 12),
            text_color="gray"
        )
        hint.pack(pady=(34, 0))

        self.result_title = ctk.CTkLabel(
            right_frame,
            text="Result",
            font=("Arial", 20, "bold")
        )
        self.result_title.pack(pady=(18, 8))

        self.pokemon_image_label = ctk.CTkLabel(
            right_frame,
            text=""
        )
        self.pokemon_image_label.pack(pady=(0, 8))

        self.result_label = ctk.CTkLabel(
            right_frame,
            text="ID: -\nEnglish: -\nJapanese: -\nChinese: -\nConfidence: -",
            font=("Arial", 14),
            justify="left",
            anchor="w"
        )
        self.result_label.pack(pady=(2, 10), padx=20, anchor="w")

        divider = ctk.CTkFrame(right_frame, height=2, width=280)
        divider.pack(pady=(2, 10))

        self.top3_label = ctk.CTkLabel(
            right_frame,
            text="Top 3:\n-",
            font=("Arial", 13),
            justify="left",
            anchor="w",
            wraplength=280
        )
        self.top3_label.pack(pady=(0, 0), padx=20, anchor="w")

    def select_audio_file(self):
        file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.wav *.ogg *.mp3"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.selected_audio_path = file_path
            self.file_label.configure(text=file_path)

    def record_audio(self):
        try:
            os.makedirs(RECORDING_DIR, exist_ok=True)

            self.file_label.configure(text="Recording... Please wait.")
            self.result_label.configure(text="Recording in progress...")
            self.top3_label.configure(text="Top 3:\n-")
            self.pokemon_image_label.configure(image=None, text="")
            self.update()

            recording = sd.rec(
                int(RECORD_SECONDS * TARGET_SR),
                samplerate=TARGET_SR,
                channels=1,
                dtype="float32"
            )

            sd.wait()

            sf.write(RECORDING_PATH, recording, TARGET_SR)

            self.selected_audio_path = RECORDING_PATH
            self.file_label.configure(text=f"Recorded:\n{RECORDING_PATH}")

            self.result_label.configure(
                text="Recording completed.\nClick Predict to identify."
            )

        except Exception as e:
            self.result_label.configure(
                text=f"Recording Error:\n{e}"
            )

    def play_audio(self):
        if not self.selected_audio_path:
            self.result_label.configure(
                text="Please select or record an audio file first."
            )
            return

        try:
            os.startfile(self.selected_audio_path)

        except Exception as e:
            self.result_label.configure(
                text=f"Audio Error:\n{e}"
            )

    def update_pokemon_image(self, pokemon_id):
        image_path = download_pokemon_image(pokemon_id)

        if image_path and os.path.exists(image_path):
            pil_image = Image.open(image_path)
            pil_image = pil_image.resize((140, 140))

            ctk_image = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=(140, 140)
            )

            self.pokemon_image_label.configure(
                image=ctk_image,
                text=""
            )

            self.pokemon_image_label.image = ctk_image

        else:
            self.pokemon_image_label.configure(
                image=None,
                text="No image available"
            )

    def show_unknown_result(self, confidence):
        self.pokemon_image_label.configure(image=None, text="Unknown")

        self.result_label.configure(
            text=(
                "ID: -\n"
                "English: Unknown\n"
                "Japanese: Unknown\n"
                "Chinese: Unknown\n"
                f"Confidence: {confidence:.2f}%\n"
                f"Status: Below threshold ({CONFIDENCE_THRESHOLD:.0f}%)"
            )
        )

        self.top3_label.configure(
            text="Top 3:\nNot shown because confidence is too low."
        )

    def predict_audio(self):
        if not self.selected_audio_path:
            self.result_label.configure(
                text="Please select or record an audio file first."
            )
            return

        try:
            X = extract_feature(self.selected_audio_path)

            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]

            label = self.encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction] * 100

            info = find_multilingual_names(label, self.name_map)

            if confidence < CONFIDENCE_THRESHOLD:
                self.show_unknown_result(confidence)
                return

            self.update_pokemon_image(info["id"])

            self.result_label.configure(
                text=(
                    f"ID: {info['id']}\n"
                    f"English: {info['en']}\n"
                    f"Japanese: {info['ja']}\n"
                    f"Chinese: {info['zh']}\n"
                    f"Confidence: {confidence:.2f}%"
                )
            )

            top3_indices = np.argsort(probabilities)[-3:][::-1]
            top3_lines = []

            for rank, idx in enumerate(top3_indices, start=1):
                top_label = self.encoder.inverse_transform([idx])[0]
                top_info = find_multilingual_names(top_label, self.name_map)
                top_conf = probabilities[idx] * 100

                top3_lines.append(
                    f"#{rank} {top_info['en']} / {top_info['zh']} - {top_conf:.2f}%"
                )

            self.top3_label.configure(
                text="Top 3:\n" + "\n".join(top3_lines)
            )

        except Exception as e:
            self.result_label.configure(
                text=f"Error:\n{e}"
            )


if __name__ == "__main__":
    app = PokemonCryApp()
    app.mainloop()