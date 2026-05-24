# Handwriting Model — מדריך אימון

מדריך זה מתאר **בדיוק** איך לאמן, להעריך ולפרוס את מודל זיהוי הכתב של AuraScribble.  
שמור את הקובץ הזה — אפשר לחזור אליו אחרי כל איסוף תיקונים מהאפליקציה.

---

## תמונה כללית

```
[אפליקציה] תיקוני משתמש → Firebase (training_data/...)
                ↓ הורדה ידנית (אופציונלי)
[מחשב] נתוני אימון (JSONL) + תיקונים → train → export ONNX
                ↓ העלאה
[Firebase] models/latest_handwriting.onnx
                ↓ OTA אוטומטי בהפעלת אפליקציה
[מכשיר] handwriting_ota.onnx
```

| מה | איפה | מתי מעדכנים |
|----|------|-------------|
| מודל ONNX (עיקרי) | Firebase `models/latest_handwriting.onnx` | **אחרי כל אימון מוצלח** |
| תיקוני משתמשים | Firebase `training_data/new/{userId}/*.json` | נאספים מהאפליקציה אוטומטית |
| מודל + vocab ב-APK | `app/src/main/assets/models/handwriting/` | רק לגרסת store / offline / שינוי vocab |

---

## התקנה חד-פעמית

פתח PowerShell מתיקיית הפרויקט:

```powershell
cd tools\handwriting-model
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

בכל פעם שפותחים טרמינל חדש לאימון:

```powershell
cd tools\handwriting-model
.\.venv\Scripts\activate
```

---

## צ'ק-ליסט — אימון מלא (העתק בכל פעם)

- [ ] **1.** יש נתונים ב-`data/processed/` (לפחות `all.jsonl` או `train.jsonl`)
- [ ] **2.** (אופציונלי) הורדת תיקונים מ-Firebase לתיקייה מקומית
- [ ] **3.** הרצת `retrain.ps1` (או שלבים ידניים למטה)
- [ ] **4.** בדיקה: `val_cer` יורד ב-`output/training_log.jsonl`
- [ ] **5.** בדיקה: `output/eval_report.json` — `cer_mean` סביר
- [ ] **6.** העלאה ל-Firebase — `upload_firebase.ps1` (אוטומטי אם יש credentials)
- [ ] **7.** (רק אם שינית vocab) עדכון `vocab.txt` ב-assets + גרסת APK
- [ ] **8.** (אופציונלי) `copy_to_android.ps1` לגיבוי ב-APK
- [ ] **9.** בדיקה במכשיר: פתיחת אפליקציה → המתנה ל-OTA → ניסוי OCR

---

## שלב 1 — הכנת נתונים

### פורמט קובץ JSONL

כל שורה = דגימה אחת:

```json
{"points": [[x, y, t], [x, y, t], ...], "text": "הטקסט הנכון", "mode": "hebrew"}
```

`mode` יכול להיות: `english`, `hebrew`, `math`, `correction`, `text`, `auto`.

שים קבצים תחת:

- `data/processed/all.jsonl` — כל הנתונים (מומלץ לפיצול)
- או ישירות `train.jsonl` / `val.jsonl` אם כבר פיצלת

### פיצול train / validation (90% / 10%)

```powershell
pwsh ./scripts/split_data.ps1
```

נוצרים:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`

> בלי `val.jsonl` האימון רץ בלי מדידת CER אמיתית ולא שומר `checkpoint_best` לפי איכות.

---

## שלב 2 — תיקונים מהאפליקציה (Firebase)

האפליקציה שומרת תיקונים ב-Firebase Storage:

- נתיב: `training_data/new/{userId}/{timestamp}.json`
- סכמה:

```json
{
  "truth": "הטקסט המתוקן",
  "prediction": "מה שהמודל חזה",
  "points": [[x, y, t], ...],
  "mode": "hebrew"
}
```

### הורדה ידנית

1. פתח [Firebase Console](https://console.firebase.google.com) → Storage  
2. הורד את תוכן `training_data/new/` (או תיקיית משתמש)  
3. שמור למשל: `data/corrections/` (קבצי `.json`)

### אימון עם תיקונים

```powershell
python src/train.py --config configs/train.yaml --corrections_dir data/corrections
```

או שלב אותם קודם ל-`all.jsonl` ואז הרץ `retrain.ps1`.

בלוג אמור להופיע: `Successfully loaded N correction samples` עם **N > 0**.

---

## שלב 3 — אימון

### אופציה א׳ — הכל בפקודה אחת (מומלץ)

```powershell
pwsh ./scripts/retrain.ps1
```

מריץ לפי הסדר:

1. `split_manifest.py` — פיצול train/val  
2. `train.py` — אימון + שמירת best לפי `val_cer`  
3. `predict.py` — חיזוי על validation  
4. `evaluate.py` — דוח CER  
5. `export_onnx.py` — `output/model.onnx` + `output/model.int8.onnx`  
6. `upload_firebase.py` — העלאה ל-Firebase (אם מוגדר service account)

### אופציה ב׳ — שלב-שלב

```powershell
pwsh ./scripts/split_data.ps1
pwsh ./scripts/train.ps1
pwsh ./scripts/predict.ps1
pwsh ./scripts/evaluate.ps1
pwsh ./scripts/export_onnx.ps1
```

### פרמטרים שימושיים

| פרמטר | דוגמה | מתי |
|--------|--------|-----|
| `--epochs` | `python src/train.py --config configs/train.yaml --epochs 3` | אימון מהיר לבדיקה |
| `--corrections_dir` | `--corrections_dir data/corrections` | אחרי הורדת תיקונים |
| `--data_path` | `--data_path data/extra.jsonl` | מיזוג קובץ נוסף |

הגדרות ב-[`configs/train.yaml`](configs/train.yaml): `learning_rate`, `epochs`, `batch_size`, `correction_loss_weight`, וכו'.

### נקודת התחלה (fine-tune)

ברירת מחדל טוענת משקולות מ:

- `models/checkpoint_best.pt`

אם הקובץ חסר — האימון מתחיל מאפס (איטי יותר, דורש הרבה נתונים).

### מה לבדוק אחרי אימון

| קובץ | מה לחפש |
|------|---------|
| `output/training_log.jsonl` | `val_cer` יורד מ-epoch ל-epoch |
| `output/checkpoint_best.pt` | נשמר רק כש-`val_cer` השתפר |
| `output/eval_report.json` | `cer_mean` — ככל שנמוך יותר, טוב יותר |
| `output/predictions.jsonl` | השוואה ידנית לדוגמאות |

---

## שלב 4 — פריסה למשתמשים (Firebase OTA)

**זה השלב העיקרי.** האפליקציה בודקת בעצמה בעת ההפעלה:

- קובץ מרוחק: `models/latest_handwriting.onnx`
- bucket: `aurascribblr.firebasestorage.app`
- מטמון מקומי: `handwriting_ota.onnx`

### העלאה אוטומטית (מומלץ)

#### הגדרה חד-פעמית

1. [Firebase Console](https://console.firebase.google.com) → Project **aurascribblr**  
2. ⚙ Project Settings → **Service accounts** → **Generate new private key**  
3. שמור את הקובץ כ:
   ```
   tools/handwriting-model/configs/firebase_service_account.json
   ```
   (אל תעלה ל-Git — הקובץ רגיש)

4. ודא שלחשבון יש הרשאת **Storage Admin** (או Firebase Admin) על ה-bucket.

#### הרצה

```powershell
pip install google-cloud-storage
pwsh ./scripts/upload_firebase.ps1
```

או אחרי pipeline מלא — `retrain.ps1` מעלה **אוטומטית** אם קיים אחד מאלה:

- `configs/firebase_service_account.json`
- משתנה סביבה `GOOGLE_APPLICATION_CREDENTIALS`
- משתנה סביבה `FIREBASE_SERVICE_ACCOUNT_JSON` (JSON מלא כ-string — מתאים ל-Kaggle Secrets)

```powershell
# דוגמה Windows — נתיב ל-SA
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\firebase-sa.json"
pwsh ./scripts/retrain.ps1
```

```powershell
python src/upload_firebase.py --local output/model.onnx --vocab configs/vocab.txt
```

הסקרipt מעלה:

| קובץ מקומי | נתיב ב-Firebase |
|------------|-----------------|
| `output/model.onnx` | `models/latest_handwriting.onnx` |
| `configs/vocab.txt` | `models/latest_vocab.txt` (לשימוש עתידי; האפליקציה כרגע טוענת vocab מ-assets) |

### העלאה ידנית (גיבוי)

```
output/model.onnx  →  Firebase Storage: models/latest_handwriting.onnx
```

דוגמאות (התאם bucket / פרויקט):

```powershell
# Firebase CLI (מהשורש של הפרויקט, אם מוגדר)
firebase storage:upload tools/handwriting-model/output/model.onnx models/latest_handwriting.onnx

# או gsutil
gsutil cp tools/handwriting-model/output/model.onnx gs://YOUR_BUCKET/models/latest_handwriting.onnx
```

או דרך Firebase Console → Storage → העלאה לנתיב `models/latest_handwriting.onnx` (החלף קובץ קיים).

### איך המשתמש מקבל עדכון

1. מפעיל את האפליקציה (Wi‑Fi / רשת)  
2. האפליקציה משווה תאריך עדכון ב-Firebase מול המקומי  
3. אם יש גרסה חדשה — מורידה ומחליפה את `handwriting_ota.onnx`  
4. **לא חייבים** APK חדש רק בשביל המודל

### בדיקה במכשיר

1. העלה ONNX חדש ל-Firebase  
2. סגור את האפליקציה לגמרי  
3. פתח מחדש (רשת פעילה)  
4. Logcat: חפש `OTA model updated`  
5. נסה OCR על כתב יד

---

## שלב 5 — עדכון assets ב-APK (לא בכל אימון)

```powershell
pwsh ./scripts/copy_to_android.ps1
```

מעתיק:

- `output/model.onnx` → `app/src/main/assets/models/handwriting/handwriting_v1.onnx`
- `configs/vocab.txt` → `app/src/main/assets/models/handwriting/vocab.txt`

**מתי כן לעשות:**

- לפני שחרור גרסה ל-Play Store  
- כשאין רשת / רוצים מודל טוב בהתקנה ראשונה  
- **חובה** אם שינית את `vocab.txt` באימון (ה-vocab לא מתעדכן ב-OTA היום)

**מתי לא חובה:**

- אחרי כל אימון — מספיק Firebase OTA למודל ONNX

---

## פתרון בעיות

| בעיה | סיבה אפשרית | פתרון |
|------|-------------|--------|
| `No samples found` | אין JSONL | צור `data/processed/all.jsonl` |
| `val_cer` לא מודפס | אין `val.jsonl` | הרץ `split_data.ps1` |
| `0 correction samples` | סכמה ישנה / נתיב שגוי | ודא שדה `truth` ב-JSON |
| OCR לא השתפר אחרי העלאה | OTA לא הסתיים | סגור/פתח אפליקציה, בדוק Logcat |
| תווים מוזרים אחרי אימון | vocab השתנה בלי APK | עדכן `vocab.txt` ב-assets |
| `ModuleNotFoundError: numpy` | venv לא פעיל | `.\.venv\Scripts\activate` |
| ONNX export נכשל | checkpoint חסר | הרץ `train.py` קודם |

---

## מבנה תיקיות

```
tools/handwriting-model/
├── configs/
│   ├── train.yaml      # היפר-פרמטרים
│   └── vocab.txt       # מילון תווים
├── data/
│   ├── processed/      # train.jsonl, val.jsonl, all.jsonl
│   └── corrections/    # תיקונים שהורדו מ-Firebase (יצירה ידנית)
├── models/
│   └── checkpoint_best.pt   # נקודת התחלה ל-fine-tune
├── output/
│   ├── checkpoint_best.pt   # הטוב ביותר לפי val_cer
│   ├── model.onnx           # לפריסה
│   ├── training_log.jsonl
│   └── eval_report.json
├── scripts/            # *.ps1
└── src/                # קוד Python
```

---

## סקריפטים — תמצית

| סקריפט | מה עושה |
|--------|---------|
| `retrain.ps1` | **הכל** — split → train → predict → evaluate → export |
| `split_data.ps1` | פיצול 90/10 |
| `train.ps1` | אימון בלבד |
| `predict.ps1` | חיזוי על val |
| `evaluate.ps1` | דוח CER |
| `export_onnx.ps1` | ייצוא ONNX |
| `upload_firebase.ps1` | **העלאה אוטומטית ל-Firebase** |
| `copy_to_android.ps1` | העתקה ל-assets (גיבוי) |

---

## פרטים טכניים (לעיון)

- **Teacher forcing**: קלט `[<bos>, t1, …]` → חיזוי `[t1, …, <eos>]` (תואם לאנדרואיד)  
- **Best checkpoint**: נשמר רק כש-`val_cer` על validation יורד  
- **תיקונים**: משקל loss גבוה יותר (`correction_loss_weight` ב-`train.yaml`)  
- **Fallback באפליקציה**: אם אין OTA → `handwriting_v1.onnx` מ-assets; אם ONNX ריק → ML Kit

---

## זרימה מקוצרת (שורה אחת לזכור)

**נתונים → `retrain.ps1` → (העלאה אוטומטית ל-Firebase) → פתח אפליקציה לבדיקה.**

---

## אימון על Kaggle (GPU)

מחברת מלאה עם כל התאים: **[KAGGLE_NOTEBOOK.md](KAGGLE_NOTEBOOK.md)**  
(העתק כל תא למחברת Kaggle, עדכן `REPO_ROOT` לנתיב ה-dataset שלך.)
