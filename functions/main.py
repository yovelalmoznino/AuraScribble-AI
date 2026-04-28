from firebase_functions import storage_fn
from firebase_admin import initialize_app, storage
import requests
import os

initialize_app()

# הגדרות גיטהאב
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_OWNER = "yovelalmoznino"
REPO_NAME = "AuraScribble-AI"
WORKFLOW_ID = "main.yml" 
THRESHOLD = 500

@storage_fn.on_object_finalized()
def trigger_training_on_threshold(event: storage_fn.CloudEvent[storage_fn.StorageObjectData]):
    file_path = event.data.name
    
    # בדיקה אם הקובץ שעלה הוא תיקון חדש
    if not file_path.startswith("training_data/new/") or not file_path.endswith(".json"):
        return

    # ספירת הקבצים בתיקייה
    bucket = storage.bucket(event.data.bucket)
    blobs = list(bucket.list_blobs(prefix="training_data/new/"))
    json_files = [b for b in blobs if b.name.endswith(".json")]
    
    count = len(json_files)
    print(f"AuraScribble: ספירת תיקונים נוכחית: {count}")

    # הפעלה רק כשמגיעים בדיוק לסף
    if count == THRESHOLD:
        print("הסף הושג! שולח פקודת אימון לגיטהאב...")
        
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_ID}/dispatches"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {GITHUB_TOKEN}",
        }
        data = {"ref": "main"} 
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 204:
            print("GitHub Action הופעל בהצלחה!")
        else:
            print(f"שגיאה בהפעלת גיטהאב: {response.status_code} - {response.text}")
