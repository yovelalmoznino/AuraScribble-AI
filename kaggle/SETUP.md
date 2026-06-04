# Kaggle Auto-Retrain — Setup Guide (חד-פעמי)

תהליך ההגדרה לוקח ~15 דקות. אחרי זה הכל אוטומטי לחלוטין.

## ארכיטקטורה

```
┌─────────────────────────────────────────────────────────────────────┐
│  GitHub Actions (יומי, חינם)                                        │
│    1. סופר תיקונים ב-Firebase                                       │
│    2. אם ≥ 500: דוחף את KAGGLE_TRAIN_V8.ipynb ל-Kaggle              │
│    3. מחכה לסיום (~75 דק׳)                                          │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Kaggle Kernel (GPU T4 חינם, ~75 דקות)                               │
│    1. קורא Firebase secret מ-Kaggle Secrets                          │
│    2. מוריד checkpoint אחרון מ-Firebase Storage                      │
│    3. מוריד תיקוני משתמשים חדשים                                     │
│    4. ממזג עם sources (IAM, MathWriting, סינתטי)                     │
│    5. מאמן V8 (CTC hybrid, 20 epochs)                                │
│    6. quality gate: אם collapse → מבטל את ה-upload                   │
│    7. מעלה ONNX + checkpoint חזרה ל-Firebase                         │
│    8. מעביר תיקונים מ-new/ ל-processed/                              │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Android App                                                         │
│    מוריד אוטומטית את latest_handwriting.onnx ב-OTA בהפעלה הבאה        │
└─────────────────────────────────────────────────────────────────────┘
```

## שלב 1 — Kaggle Dataset (10 דקות)

ה-Bundle (קוד + 270K דגימות) צריך להיות זמין ל-Kaggle Kernel.

1. דחסי את `colab_bundle/` ל-`aurascribble-bundle.zip` (כמו ש-7-zipped אתמול)
2. ב-Kaggle:
   - לחצי על **Datasets → New Dataset**
   - Title: `aurascribble-bundle`
   - URL slug יהיה אוטומטית `yourusername/aurascribble-bundle`
   - העלי את הזיפ (Kaggle יחלץ אוטומטית)
   - Visibility: **Private**
   - Create

3. **שמרי את ה-slug** (`yourusername/aurascribble-bundle`) — נצטרך אותו בשלב 4

## שלב 2 — Kaggle API key (2 דקות)

1. https://www.kaggle.com/settings → גלגלי ל-**API** → **Create New API Token**
2. ייפול קובץ `kaggle.json` — פתחי אותו, יש שם:
   ```json
   {"username": "yourusername", "key": "abc123..."}
   ```
3. שמרי את שני הערכים האלה לשלב 4

## שלב 3 — Kaggle Kernel ראשון (3 דקות)

צריך ליצור kernel ריק פעם אחת כדי שיהיה לו slug ב-Kaggle. ה-GitHub Actions יעדכן את התוכן שלו אוטומטית בכל ריצה.

1. ב-Kaggle: **Code → New Notebook**
2. Title: `aurascribble-train` (או כל שם)
3. בתא הראשון, הדביקי:
   ```python
   print("placeholder — GitHub Actions will overwrite this")
   ```
4. Settings (גלגל ימני):
   - Accelerator: **GPU T4 x1**
   - Internet: **On**
   - Persistence: **No files persisted between runs**
5. **Save Version → Save & Run All** (זה ייצור את ה-slug)
6. אחרי שזה רץ — שמרי את ה-slug שלך (משהו כמו `yourusername/aurascribble-train`)

## שלב 4 — Kaggle Secrets (1 דקה)

ב-Kaggle Notebook שיצרת (`aurascribble-train`):

1. **Add-ons → Secrets**
2. **Add a new secret**:
   - Label: `FIREBASE_SERVICE_ACCOUNT_JSON`
   - Value: כל ה-JSON של Firebase service account (התחלה: `{"type": "service_account", ...}`)
3. Save

## שלב 5 — GitHub Secrets & Variables (2 דקות)

ב-GitHub repository, **Settings → Secrets and variables → Actions**:

### Secrets (סודיים):
| Name | Value |
|------|-------|
| `FIREBASE_SERVICE_ACCOUNT_JSON` | כל ה-JSON של Firebase service account |
| `KAGGLE_USERNAME` | מ-`kaggle.json` |
| `KAGGLE_KEY` | מ-`kaggle.json` |

### Variables (לא סודיים, אבל קל לעדכן):
| Name | Value |
|------|-------|
| `KAGGLE_KERNEL_SLUG` | `yourusername/aurascribble-train` (משלב 3) |
| `KAGGLE_DATASET_SLUG` | `yourusername/aurascribble-bundle` (משלב 1) |

## שלב 6 — ניסיון ראשון (5 דקות)

1. ב-GitHub: **Actions → AuraScribble Auto-Retrain → Run workflow**
2. סמני **Force retrain** (כדי לעקוף את ה-threshold)
3. Run workflow

תוכלי לעקוב בזמן אמת. הריצה תיקח ~80-100 דקות סך הכל:
- ~3 דקות GitHub Actions (push notebook + wait)
- ~75 דקות Kaggle (train)
- ~2 דקות upload + archive

אחרי שזה נגמר בהצלחה:
- בדקי את `result.json` ב-job summary
- בדקי שיש קובץ חדש ב-Firebase Storage: `models/latest_handwriting.onnx`
- האפליקציה תוריד אותו אוטומטית בהפעלה הבאה (OTA)

## שלב 7 — לוודא ש-Cron עובד

ה-workflow רץ **כל יום ב-06:00 UTC** (≈ 09:00 שעון ישראל). אם פחות מ-500 תיקונים — מדלג בלי לבזבז משאבי Kaggle.

צריך לוודא רק שיש לפחות commit אחד ב-default branch בחודש האחרון (GitHub משבית cron של repos לא פעילים).

## תחזוקה שוטפת

**בדרך כלל אין מה לעשות.** הדברים היחידים שיכולים לדרוש התערבות:

1. **תיקון bug בקוד** — push ל-repo. GitHub Actions תופס את הגרסה החדשה של ה-notebook אוטומטית.
2. **עדכון bundle** (הוספת דאטה חדשה גדולה) — תעלי ל-Kaggle Dataset גרסה חדשה, אותו slug, "New Version".
3. **שינוי threshold** — עדכני את `RETRAIN_THRESHOLD` ב-`main.yml`.
4. **debug ריצה כושלת** — Actions → run לאחרון → Logs. אם בעיית Kaggle — `kaggle kernels output yourusername/aurascribble-train`.

## עלות

- Kaggle: **0 ש"ח** (30 שעות GPU/שבוע, ריצה ~1.25 שעה — שימוש זניח)
- GitHub Actions: **0 ש"ח** (2000 דקות חינם/חודש לפרטיים, ריצה משתמשת ~3 דקות)
- Firebase Storage: **כמעט 0** (ONNX קטן, ~20MB)

**סה"כ: 0 ש"ח לתחזוקה אוטומטית מלאה.**
