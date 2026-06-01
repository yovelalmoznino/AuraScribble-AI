# הרצת Kaggle v3.1 — מה לעשות (עברית)

## לפני Kaggle (מחשב)

```powershell
cd C:\Users\yalmo\Desktop\Notebbok\tools\handwriting-model
python src\pack_kaggle_zip.py --output handwriting_training_bundle.zip
```

העלה ZIP חדש ל-Kaggle dataset.

---

## במחברת — תאים שחייבים עדכון

| תא | פעולה |
|----|--------|
| **10b** | מחולל short + medium + iam_long (לא רק short) |
| **11** | config v3.1 |
| **12** | phase 1 — קצר (כמו קודם) |
| **12b** | phase 2a — **בינוני** (33–72 תווים) |
| **12c** | phase 2b — **מלא** (patience 12, lr נמוך) |
| **12d** | phase 2c — **רק IAM ארוך** |
| **13** | sanity עם `decode_quality` (תופס `the and`, `ה`, `\frac{1}`) |
| **15** | Firebase רק אם `passes_export_gate` |

תאים 1–9, 14, 16 — כמו במדריך `KAGGLE_NOTEBOOK_REBUILD.md`.

---

## סדר הרצה

```
1→9 → 10 → 10b → 11 → 12 → 12b → 12c → 12d → 13 → (14–15 רק אם עבר)
```

זמן משוער: ~2–3 שעות (תלוי ב-GPU).

---

## מתי מעלים לאפליקציה

בתא 13:

- **COLLAPSE ≤ 2** מתוך 24 (לא רק `[OK]` ישן)
- `val_cer < 0.5` ב-checkpoint
- אין רוב דגימות `the and` / `ה` בודדת

בתא 15: `cer_mean ≤ 0.35` ו-collapse rate נמוך.

---

## אם רואים `wan theroulis` / `val_cer` ~0.85 אבל פלט גרוע

זה **לא מוכן** — CER נמוך כי הפלט **קצר**, לא כי נכון.  
ראה [`HANDWRITING_REALITY.md`](HANDWRITING_REALITY.md).

**אל תריץ עוד אימון על פסקאות IAM.** הצעד הבא בפרויקט: **אימון שורה-שורה** (max ~64 תווים).

## אם שוב נכשל על IAM

- יותר נתוני IAM **מפוצלים לשורות**
- ML Kit לאנגלית ארוכה באפליקציה (כבר קיים כ-fallback)

Decode משופר (`decode_quality`) עוזר ב-Python — **לא** ב-ONNX באנדרואיד עד שנוסיף אותו גם ב-Kotlin.
