import firebase_admin
from firebase_admin import credentials, storage
import os
import json
from pathlib import Path

# 1. הגדרות התחברות
# 1. הגדרות התחברות
cred = credentials.Certificate("firebase-service-account.json")
firebase_admin.initialize_app(cred) # בלי הגדרות נוספות כאן
bucket = storage.bucket("aurascribblr.firebasestorage.app") # השם המדויק כאן
def download_data():
    """מוריד את התיקונים החדשים מהאפליקציה"""
    Path("data/new_samples").mkdir(parents=True, exist_ok=True)
    blobs = bucket.list_blobs(prefix='training_data/new/')
    count = 0
    for blob in blobs:
        filename = blob.name.split('/')[-1]
        blob.download_to_filename(f"data/new_samples/{filename}")
        # העברה לתיקיית processed כדי לא להתאמן פעמיים
        bucket.rename_blob(blob, f"training_data/processed/{filename}")
        count += 1
    return count

def run_training():
    """מפעיל את קוד האימון המעודכן"""
    print("Starting fine-tuning...")
    # יצירת תיקיית פלט כדי שלא תהיה שגיאת נתיב
    os.makedirs("output", exist_ok=True)
    
    # הרצה עם הגדרת output_dir מפורשת בתוך הפקודה
    # אנחנו מוסיפים דגל שיגרום ל-train.py להשתמש בתיקייה שיצרנו
    cmd = "python src/train.py --config configs/train.yaml --corrections_dir data/new_samples"
    os.system(cmd)
    
def upload_model():
    """מעלה את המודל המשופר חזרה לענן"""
    # הנתיב צריך להיות output/model.onnx (או השם שמוגדר ב-export)
    model_path = "output/latest_model.onnx" 
    if os.path.exists(model_path):
        blob = bucket.blob('models/latest_handwriting.onnx')
        blob.upload_from_filename(model_path)
        print("New model uploaded to Firebase!")
    else:
        print(f"Model file not found at {model_path}")


if __name__ == "__main__":
    new_data_count = download_data()
    if new_data_count > 0:
        run_training()
        upload_model()
    else:
        print("No new corrections found. System is up to date.")
