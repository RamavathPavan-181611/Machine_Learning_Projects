import gdown
import zipfile
import os

def download_and_extract(file_id, output_zip, extract_to):
    url = f"https://drive.google.com/uc?id={file_id}"

    # Skip download if already extracted
    if os.path.exists(extract_to) and all(
        os.path.exists(os.path.join(extract_to, lang, "model.pkl")) and
        os.path.exists(os.path.join(extract_to, lang, "vectorizer.pkl"))
        for lang in ["English", "Gujarati", "Hindi", "Multilingual"]
    ):
        print("Models and vectorizers already present.")
        return

    print("Downloading ZIP...")
    gdown.download(url, output_zip, quiet=False)

    print("Extracting ZIP...")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(output_zip)
    print("Done.")


