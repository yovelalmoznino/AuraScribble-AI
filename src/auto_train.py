import firebase_admin
from firebase_admin import credentials, storage
import os
import json
from pathlib import Path

# 1. הגדרות התחברות
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-service-account.json")
    firebase_admin.initialize_app(cred)

BUCKET_NAME = "aurascribblr.firebasestorage.app"
bucket = storage.bucket(BUCKET_NAME)

def process_and_merge_data():
    """
    מורידה תיקונים חדשים, ממזגת אותם לקובץ Master אחד,
    מעלה את המאסטר המעודכן לענן ומוחקת את הקבצים הבודדים.
    """
    Path("data").mkdir(exist_ok=True)
    master_path = "data/master_corrections.jsonl"
    remote_master_path = "data/master_corrections.jsonl"
    
    # --- 1. הורדת המאסטר הקיים ---
    master_content = ""
    blob_master = bucket.blob(remote_master_path)
    if blob_master.exists():
        master_content = blob_master.download_as_text().strip()
        print("Existing master data downloaded.")

    # --- 2. איסוף תיקונים חדשים ---
    blobs = list(bucket.list_blobs(prefix='training_data/new/'))
    new_entries = []
    
    for blob in blobs:
        if blob.name.endswith('/') or not blob.name.endswith('.json'):
            continue
            
        content = blob.download_as_text().strip()
        new_entries.append(content)
        
        # העברה לתיקיית processed (ארכיון בענן)
        new_path = blob.name.replace('training_data/new/', 'training_data/processed/')
        bucket.rename_blob(blob, new_path)
        print(f"Processed: {blob.name}")

    if not new_entries:
        return 0, master_path

    # --- 3. מיזוג ושמירה מקומית ---
    # מוודא שכל שורה היא JSON תקין ומחבר למאסטר
    combined_content = master_content
    if combined_content:
        combined_content += "\n"
    combined_content += "\n".join(new_entries)

    with open(master_path, 'w', encoding='utf-8') as f:
        f.write(combined_content.strip())

    # --- 4. עדכון המאסטר בענן (לשימוש עתידי בקאגל) ---
    blob_master.upload_from_filename(master_path)
    print(f"Success: Master file updated with {len(new_entries)} new corrections.")
    
    return len(new_entries), master_path

def run_training(data_path):
    """מפעיל את קוד האימון על קובץ המאסטר המאוחד"""
    print(f"Starting fine-tuning using {data_path}...")
    os.makedirs("output", exist_ok=True)
    
    # עדכון פקודת האימון שתשתמש בקובץ ה-jsonl המאוחד במקום בתיקייה
    cmd = f"python src/train.py --config configs/train.yaml --data_path {data_path} --epochs 5"
    os.system(cmd)
    
def upload_model():
    """מעלה את המודל המשופר חזרה לענן"""
    model_path = "output/latest_model.onnx" 
    if os.path.exists(model_path):
        blob = bucket.blob('models/latest_handwriting.onnx')
        blob.upload_from_filename(model_path)
        print("New model uploaded to Firebase!")
    else:
        print(f"Model file not found at {model_path}")

def download_base_model():
    """מוריד את ה-Weights המקוריים (PyTorch) להתחלת Fine-tuning"""
    Path("models").mkdir(exist_ok=True)
    blob = bucket.blob('models/checkpoint_best.pt')
    
    if blob.exists():
        blob.download_to_filename("models/checkpoint_best.pt")
        print("Base model weights downloaded.")
    else:
        print("Warning: No base model weights found.")

if __name__ == "__main__":
    # 1. עיבוד ומיזוג נתונים
    new_count, local_data_file = process_and_merge_data()
    
    if new_count > 0:
        # 2. הכנת המודל
        download_base_model()
        # 3. אימון (Fine-tuning)
        run_training(local_data_file)
        # 4. העלאת תוצאה
        upload_model()
    else:
        print("No new corrections to process. System is up to date.")
