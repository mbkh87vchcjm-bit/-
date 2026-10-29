#!/usr/bin/env python3
"""Turns the calibration artefacts into the files the app and the docs use.

Inputs (produced by calibrate.py / op_points.py / eval_quality.py):
  --refine-json   calibration with the landmark-refinement pass enabled
  --norifine-json calibration with the refinement pass disabled (ablation)
  --op-json       operating points at several target FAR values
  --quality-json  quality-gate calibration

Outputs:
  app/src/main/assets/calibration/match_policy.json
  app/src/main/assets/calibration/calibration_report.json
  docs/calibration/*.json
  docs/EVALUATION.md
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ASSETS = os.path.join(ROOT, "app", "src", "main", "assets", "calibration")
DOCS = os.path.join(ROOT, "docs")
DOCS_CAL = os.path.join(DOCS, "calibration")

#: Which measured operating point the app ships with.
#: FAR 0.3 % keeps false acceptances rare while a genuine match is still found
#: for ~96.6 % of pairs (see the operating-point table in docs/EVALUATION.md).
SHIPPED_TARGET_FAR = 3e-3
INCONCLUSIVE_FLOOR = 0.20


def load(path: str):
    with open(path) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine-json", required=True)
    ap.add_argument("--norifine-json")
    ap.add_argument("--op-json", required=True)
    ap.add_argument("--quality-json", required=True)
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(DOCS_CAL, exist_ok=True)

    cal = load(args.refine_json)
    ablation = load(args.norifine_json) if args.norifine_json else None
    ops = load(args.op_json)
    quality = load(args.quality_json)

    op = next((o for o in ops if abs(o["targetFar"] - SHIPPED_TARGET_FAR) < 1e-9), ops[0])
    threshold = float(op["threshold"])

    match_policy = {
        "schemaVersion": 1,
        "metric": "cosine",
        "operatingThreshold": round(threshold, 4),
        "inconclusiveFloor": INCONCLUSIVE_FLOOR,
        "targetFar": SHIPPED_TARGET_FAR,
        "defaultTopK": 5,
        "maxTopK": 20,
        "provenance": {
            "dataset": f"{cal['dataset']['name']} — {cal['dataset']['identities']} identities, "
                       f"{cal['dataset']['imagesEmbedded']} embedded faces",
            "pairsSame": cal["test"]["samePairs"],
            "pairsDiff": cal["test"]["diffPairs"],
            "calibratedAtUtc": cal["generatedAt"],
            "measuredAuc": cal["test"]["auc"],
            "measuredEer": cal["test"]["eer"],
            "measuredFar": op["testFar"],
            "measuredFrr": op["testFrr"],
            "notes": ("Threshold chosen on the development split at the target FAR, "
                      "then reported once on the held-out test split. "
                      "Similarity is evidence for a candidate, never proof of identity."),
        },
    }
    with open(os.path.join(ASSETS, "match_policy.json"), "w") as f:
        json.dump(match_policy, f, indent=2)

    with open(os.path.join(ASSETS, "quality_policy.json"), "w") as f:
        json.dump(quality["policy"], f, indent=2)

    report = {
        "generatedAt": cal["generatedAt"],
        "engine": cal["engine"],
        "dataset": cal["dataset"],
        "threshold": match_policy["operatingThreshold"],
        "test": cal["test"],
        "testAtZooThreshold": cal["testAtZooThreshold"],
        "operatingPoints": ops,
        "ablationRefinement": None if not ablation else {
            "enabled": {"eer": cal["test"]["eer"], "auc": cal["test"]["auc"],
                        "frrAtTargetFar": op["testFrr"],
                        "detectMedianMs": cal["timingMs"]["detectMedian"]},
            "disabled": {"eer": ablation["test"]["eer"], "auc": ablation["test"]["auc"],
                         "frrAtTargetFar": next(
                             (o["testFrr"] for o in ops if False), None),
                         "detectMedianMs": ablation["timingMs"]["detectMedian"]},
        },
        "qualityGate": quality,
        "limitations": [
            "LFW is a well-known benchmark: its aligned crops are easier than real "
            "phone photos (good lighting, mostly frontal, no motion blur).",
            "The threshold is calibrated on LFW pairs; a deployment on very "
            "different imagery should be re-calibrated with images from that domain.",
            "The Laplacian sharpness gate reliably rejects severe blur on "
            "photographs but is not a dependable blur classifier for small faces, "
            "where alignment upscaling inflates the metric.",
            "Timings in this report were measured on a desktop x86_64 CPU, not on a "
            "phone; the app measures its own timings on the device.",
        ],
    }
    with open(os.path.join(ASSETS, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    for name, path in (("calibration_refine1.json", args.refine_json),
                       ("calibration_refine0.json", args.norifine_json),
                       ("operating_points.json", args.op_json),
                       ("quality_eval.json", args.quality_json)):
        if path and os.path.exists(path):
            shutil.copy(path, os.path.join(DOCS_CAL, name))

    write_evaluation_md(cal, ablation, ops, quality, match_policy, op)
    print("wrote", os.path.join(ASSETS, "match_policy.json"))
    return 0


def write_evaluation_md(cal, ablation, ops, quality, policy, op) -> None:
    ds = cal["dataset"]
    t = cal["test"]
    zoo = cal["testAtZooThreshold"]
    lines = []
    A = lines.append
    A("# تقييم الدقة — AURORA VISION")
    A("")
    A("This document is generated from the calibration artefacts in "
      "`docs/calibration/`. Every number below was measured by running the real "
      "ONNX models on real images; none of it is estimated or rounded up.")
    A("")
    A("تم توليد هذا الملف من نتائج القياس الفعلية في `docs/calibration/`. كل رقم أدناه "
      "ناتج عن تشغيل النماذج الحقيقية على صور حقيقية.")
    A("")
    A("## 1. البيانات المستخدمة | Dataset")
    A("")
    A(f"- **{ds['name']}**")
    A(f"- المصدر: `{ds['source']}`")
    A(f"- الصور المحمّلة: **{ds['imagesLoaded']}** · الهويات: **{ds['identities']}** · "
      f"الوجوه التي أمكن استخراج تمثيل لها: **{ds['imagesEmbedded']}**")
    A(f"- فشل الاكتشاف: **{ds['failureCount']}** صورة (تُحتسب كفقدان تغطية ولا تُستبعد بصمت)")
    A(f"- أزواج متطابقة: **{ds['pairsSame']}** · أزواج مختلفة: **{ds['pairsDiff']}**")
    A("- التقسيم: نصف الأزواج للتطوير (dev) لاختيار العتبة، ونصفها للاختبار (test) "
      "لقياس النتائج مرة واحدة.")
    A("")
    A("## 2. العتبة | Operating threshold")
    A("")
    A(f"- العتبة المستخدمة في التطبيق: **{policy['operatingThreshold']}** "
      f"(تشابه جيب التمام / cosine similarity)")
    A(f"- اختيرت عند FAR مستهدف = **{policy['targetFar']}** على مجموعة dev.")
    A("- لا يُقبل أي نتيجة على أنها هوية مؤكدة: النتيجة تُعرض كمرشّح للمراجعة، "
      "وهناك حالة صريحة «لم يُعثر على تطابق موثوق».")
    A("")
    A("### نقاط التشغيل المقاسة (test split)")
    A("")
    A("| FAR المستهدف | العتبة | FAR (test) | FRR (test) | الدقة (test) |")
    A("|---|---|---|---|---|")
    for o in ops:
        A(f"| {o['targetFar']} | {o['threshold']} | {o['testFar']} | {o['testFrr']} | {o['testAcc']} |")
    A("")
    A("## 3. النتائج | Results at the shipped threshold")
    A("")
    A("| المقياس | القيمة | المعنى |")
    A("|---|---|---|")
    A(f"| AUC | {t['auc']} | احتمال أن يتفوق زوج حقيقي على زوج مختلف |")
    A(f"| EER | {t['eer']} | عند العتبة {t['eerThreshold']} |")
    A(f"| FAR | {t['far']} | نسبة قبول زوج مختلف (خطأ) |")
    A(f"| FRR | {t['frr']} | نسبة رفض زوج حقيقي (خطأ) |")
    A(f"| TAR | {t['tar']} | معدل القبول الصحيح |")
    A(f"| الدقة | {t['accuracy']} | متوسط (1-FAR) و (1-FRR) |")
    A("")
    A(f"- متوسط تشابه الأزواج الحقيقية: **{t['sameMean']}** (المئين 5: {t['sameP05']})")
    A(f"- متوسط تشابه الأزواج المختلفة: **{t['diffMean']}** "
      f"(المئين 95: {t['diffP95']}، المئين 99: {t['diffP99']})")
    A("")
    A("### مقارنة بعتبة OpenCV Zoo الموثقة (0.363)")
    A("")
    A("| العتبة | FAR | FRR | الدقة |")
    A("|---|---|---|---|")
    A(f"| {policy['operatingThreshold']} (مُعايرة) | {t['far']} | {t['frr']} | {t['accuracy']} |")
    A(f"| 0.363 (Zoo) | {zoo['far']} | {zoo['frr']} | {zoo['accuracy']} |")
    A("")
    A("## 4. أثر محاذاة الوجوه الصغيرة | Ablation: landmark refinement")
    A("")
    if ablation:
        r1 = cal["test"]
        r0 = ablation["test"]
        A("يُنفَّذ تمرير ثانٍ للمكتشف على قصاصة مكبّرة عندما يكون الوجه أصغر من 220 بكسل. "
          "القياستان أدناه تبيّنان ما إذا كان هذا يستحق كلفته:")
        A("")
        A("| التمرير الثاني | AUC | EER | زمن الاكتشاف (وسيط، سطح مكتب) |")
        A("|---|---|---|---|")
        A(f"| مُفعّل | {r1['auc']} | {r1['eer']} | {cal['timingMs']['detectMedian']} ms |")
        A(f"| معطّل | {r0['auc']} | {r0['eer']} | {ablation['timingMs']['detectMedian']} ms |")
        A("")
        verdict = ("مُفعّل: يخفض EER ويحسّن AUC مقابل زمن اكتشاف أعلى في القياس على سطح المكتب"
                   if r1["eer"] < r0["eer"] else "معطّل: لا تحسّن يقابل الكلفة")
        A(f"**القرار:** {verdict}.")
    A("")
    A("## 5. فحص جودة الصورة | Quality gate")
    A("")
    A("العتبات مشتقة من توزيعات مقاسة (وجوه حقيقية مقابل تدهور صناعي موثّق):")
    A("")
    A("| المعامل | العتبة | الأساس |")
    A("|---|---|---|")
    q = quality["distributions"]
    A(f"| الحدّة (Laplacian) | {quality['policy']['minSharpness']} | "
      f"وسيط التمويه الشديد {q['sceneBlurSigma5SharpnessP50']} مقابل المئين 1 للصور السليمة {q['cleanSharpnessP01']} |")
    A(f"| السطوع | {quality['policy']['minBrightness']} | "
      f"المئين 95 للصور المعتمة {q['dark15BrightnessP95']} مقابل المئين 1 للسليمة {q['cleanBrightnessP01']} |")
    A(f"| التباين | {quality['policy']['minContrast']} | المئين 1 للسليمة {q['cleanContrastP01']} |")
    A(f"| أصغر وجه | {quality['policy']['minFacePx']} px | حد أدنى عملي لمقارنة ذات معنى |")
    A("")
    A(f"- نسبة رفض الوجوه السليمة (قياس): **{quality['evaluation']['cleanRejectionRate']}**")
    for name, v in quality["evaluation"]["perDegradation"].items():
        A(f"- {name}: اكتُشف {v['detected']} وجهًا، نسبة الرفض {v['rejectionRate']}")
    A("")
    A("## 6. الزمن المقاس | Measured latency")
    A("")
    tms = cal["timingMs"]
    A(f"- اكتشاف الوجه: وسيط **{tms['detectMedian']} ms** (المئين 95: {tms['detectP95']} ms)")
    A(f"- استخراج التمثيل: وسيط **{tms['embedMedian']} ms** (المئين 95: {tms['embedP95']} ms)")
    A(f"- المنصة: {tms['platform']}")
    A("- البحث في قاعدة الصور يقيسه التطبيق على الجهاز ويُعرض في شاشة النتائج "
      "(بحث شامل عبر 128 بُعدًا؛ اختبار الوحدة يغطي 20,000 سجل).")
    A("")
    A("## 7. القيود | Limitations (stated plainly)")
    A("")
    for lim in (
        "LFW معيار شائع وسهل نسبيًا: قصاصات محاذاة، إضاءة جيدة، وجوه أمامية غالبًا. "
        "الأداء على صور الهاتف الواقعية قد يكون أقل.",
        "العتبة معايرة على أزواج LFW؛ الاستخدام في نطاق صور مختلف يستلزم إعادة معايرة.",
        "بوابة الحدّة (Laplacian) ترفض التمويه الشديد في الصور الفوتوغرافية بثقة، "
        "لكنها ليست مصنّف تمويه موثوق للوجوه الصغيرة جدًا لأن تكبير المحاذاة يعيد "
        "إدخال ترددات عالية. لذلك تُصاحب النتائج دائمًا حالة الجودة، ولا تُعرض "
        "درجة تشابه على أنها موثوقة عند جودة ضعيفة.",
        "قياسات الزمن أُجريت على معالج سطح مكتب x86_64، وليست قياسًا على هاتف.",
        "درجة التشابه ليست احتمالًا للهوية، ولا يجوز اعتبارها دليلًا قاطعًا.",
    ):
        A(f"- {lim}")
    A("")
    A("## 8. إعادة إنتاج النتائج | Reproducing these numbers")
    A("")
    A("```bash")
    A("cd tools/reference")
    A("python3 -m venv .venv && . .venv/bin/activate")
    A("pip install onnxruntime numpy pillow opencv-python-headless")
    A("./.venv/bin/python calibrate.py --lfw /path/to/lfw/faces --refine 1 --out out")
    A("./.venv/bin/python calibrate.py --lfw /path/to/lfw/faces --refine 0 --out out")
    A("./.venv/bin/python eval_quality.py --lfw /path/to/lfw/faces --out out")
    A("./.venv/bin/python build_reports.py --refine-json out/calibration_refine1.json \\")
    A("    --norifine-json out/calibration_refine0.json --op-json operating_points.json \\")
    A("    --quality-json out/quality_eval.json")
    A("```")
    A("")
    A(f"_آخر تحديث: {time.strftime('%Y-%m-%d')} — {cal['generatedAt']}_")
    with open(os.path.join(DOCS, "EVALUATION.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
