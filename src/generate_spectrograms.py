import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data"
OUTPUT_DIR = "spectrograms"

TARGET_SR = 16000

IMG_SIZE = (3, 3)


def generate_mel_spectrogram(audio_path, output_path):
    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR)

        y, _ = librosa.effects.trim(y, top_db=25)

        if len(y) == 0:
            return

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            fmax=8000
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=IMG_SIZE)

        librosa.display.specshow(
            mel_db,
            sr=sr,
            x_axis=None,
            y_axis=None,
            cmap="magma"
        )

        plt.axis("off")

        plt.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()

    except Exception as e:
        print(f"[Error] {audio_path}: {e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pokemon_folders = sorted([
        folder for folder in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, folder))
    ])

    total_count = 0

    for folder_name in pokemon_folders:
        input_folder = os.path.join(DATA_DIR, folder_name)
        output_folder = os.path.join(OUTPUT_DIR, folder_name)

        os.makedirs(output_folder, exist_ok=True)

        audio_files = [
            f for f in os.listdir(input_folder)
            if f.endswith(".wav")
        ]

        for audio_file in audio_files:
            input_path = os.path.join(input_folder, audio_file)

            output_name = os.path.splitext(audio_file)[0] + ".png"
            output_path = os.path.join(output_folder, output_name)

            if os.path.exists(output_path):
                continue

            generate_mel_spectrogram(
                input_path,
                output_path
            )

            total_count += 1

        print(f"[Done] {folder_name}")

    print("\nSpectrogram generation completed.")
    print(f"Total generated: {total_count}")


if __name__ == "__main__":
    main()