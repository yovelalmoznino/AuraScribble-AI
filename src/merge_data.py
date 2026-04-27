import firebase_admin
from firebase_admin import credentials, storage
import json
import os

def merge_and_cleanup():
    # אתחול (השתמשי ב-Service Account שכבר מוגדר לך בגיטהאב)
    if not firebase_admin._apps:
        cred = credentials.Certificate('service-account.json')
        firebase_admin.initialize_app(cred, {'storageBucket': 'your-app.appspot.com'})

    bucket = storage.bucket()
    
    # 1. הורדת ה-Master הקיים (אם יש כזה)
    master_blob = bucket.blob('data/master_corrections.jsonl')
    master_data = ""
    if master_blob.exists():
        master_data = master_blob.download_as_text()

    # 2. איסוף כל התיקונים החדשים מתיקיית ה-corrections
    new_corrections = []
    blobs = list(bucket.list_blobs(prefix='corrections/'))
    
    if not blobs:
        print("לא נמצאו תיקונים חדשים למיזוג.")
        return

    for blob in blobs:
        if blob.name.endswith('.json') or blob.name.endswith('.jsonl'):
            content = blob.download_as_text()
            new_corrections.append(content.strip())

    # 3. איחוד: Master ישן + תיקונים חדשים
    updated_master = master_data.strip() + "\n" + "\n".join(new_corrections)
    
    # 4. העלאת ה-Master המעודכן חזרה
    master_blob.upload_from_string(updated_master.strip(), content_type='application/octet-stream')
    print(f"מיזגתי {len(new_corrections)} תיקונים לתוך המאסטר.")

    # 5. ניקוי: מחיקת הקבצים הבודדים כדי לאפס את המונה
    for blob in blobs:
        blob.delete()
    print("תיקיית התיקונים נוקתה. המונה התאפס.")

if __name__ == "__main__":
    merge_and_cleanup()
