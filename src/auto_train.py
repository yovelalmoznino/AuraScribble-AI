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
    """מוריד את התיקונים החדשים מכל תיקיות המשנה בתוך new"""
    current_bucket = storage.bucket("aurascribblr.firebasestorage.app")
    Path("data/new_samples").mkdir(parents=True, exist_ok=True)
    
    # prefix='training_data/new/' יביא את כל מה שמתחיל בנתיב הזה
    blobs = current_bucket.list_blobs(prefix='training_data/new/')
    count = 0
    
    for blob in blobs:
        # בדיקה שזה קובץ ולא תיקייה (blobs שמסתיימים ב-/ הם תיקיות)
        if blob.name.endswith('/'):
            continue
            
        # יצירת שם קובץ ייחודי כדי שלא יהיו התנגשויות (מחליפים סלאשים בקו תחתון)
        safe_filename = blob.name.replace('/', '_')
        blob.download_to_filename(f"data/new_samples/{safe_filename}")
        
        # העברה לתיקיית processed
        new_path = blob.name.replace('training_data/new/', 'training_data/processed/')
        current_bucket.rename_blob(blob, new_path)
        
        count += 1
        print(f"Downloaded: {blob.name}")
        
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

def download_base_model():
    """מוריד את המודל המקורי מ-Firebase כדי שיהיה על מה להתאמן"""
    current_bucket = storage.bucket("aurascribblr.firebasestorage.app")
    Path("models").mkdir(exist_ok=True)
    
    # שימי לב שהשם כאן חייב להתאים לשם שהעלית ל-Firebase
    blob = current_bucket.blob('models/base_model.pth')
    
    if blob.exists():
        print("Downloading base model from Firebase...")
        blob.download_to_filename("models/base_model.pth")
        print("Base model downloaded successfully.")
    else:
        print("Warning: No base model found in Firebase. Training from scratch might fail.")



if __name__ == "__main__":
    new_data_count = download_data()
    download_base_model()
    if new_data_count > 0:
        run_training()
        upload_model()
    else:
        print("No new corrections found. System is up to date.")
