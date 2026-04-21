import kagglehub
import os
import shutil
import glob

DEST_DIR = "/app/data"
EXPECTED_FILES = ["fraudTrain.csv", "fraudTest.csv"]


def already_downloaded():
    return all(
        os.path.isfile(os.path.join(DEST_DIR, f)) for f in EXPECTED_FILES
    )

def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    if already_downloaded():
        print("Le dataset est déjà téléchargé dans", DEST_DIR)
        return

    print("Téléchargement du dataset")
    path = kagglehub.dataset_download("kartik2112/fraud-detection")

    for csv_file in glob.glob(os.path.join(path, "*.csv")):
        dest = os.path.join(DEST_DIR, os.path.basename(csv_file))
        shutil.copy2(csv_file, dest)
    print("Dataset prêt dans", DEST_DIR)

if __name__ == "__main__":
    main()
