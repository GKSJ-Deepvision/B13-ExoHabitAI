#!/usr/bin/env python3
"""
ExoHabitAI — Assets Folder Setup Script
========================================
Run this from your project root (B13-EXOHABITAI/) to:
  1. Create the assets/images/ folder structure
  2. Copy all report PNGs into assets/images/ automatically
  3. Print a checklist of screenshots you still need to take manually
"""

import os
import shutil
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).parent          # folder where this script lives
REPORTS_DIR    = PROJECT_ROOT / "reports"
ASSETS_DIR     = PROJECT_ROOT / "assets" / "images"

# ─── Filenames that must match README.md exactly ───────────────────────────────

AUTO_COPY = {
    # source (in reports/)                         : dest (in assets/images/)
    "missing_values_heatmap.png"                  : "missing_values_heatmap.png",
    "model_comparison.png"                        : "model_comparison.png",
    "feature_importance.png"                      : "feature_importance.png",
    "Final Selected Model_confusion_matrix.png"   : "Final_Selected_Model_confusion_matrix.png",
    "Final Selected Model_roc_curve.png"          : "Final_Selected_Model_roc_curve.png",
    "Tuned XGBoost_confusion_matrix.png"          : "Tuned_XGBoost_confusion_matrix.png",
    "Tuned XGBoost_roc_curve.png"                 : "Tuned_XGBoost_roc_curve.png",
    "Random Forest_confusion_matrix.png"          : "Random_Forest_confusion_matrix.png",
    "Random Forest_roc_curve.png"                 : "Random_Forest_roc_curve.png",
    "Logistic Regression_confusion_matrix.png"    : "Logistic_Regression_confusion_matrix.png",
    "Logistic Regression_roc_curve.png"           : "Logistic_Regression_roc_curve.png",
    "Decision Tree_confusion_matrix.png"          : "Decision_Tree_confusion_matrix.png",
    "Decision Tree_roc_curve.png"                 : "Decision_Tree_roc_curve.png",
    "Tuned Random Forest_confusion_matrix.png"    : "Tuned_Random_Forest_confusion_matrix.png",
    "Tuned Random Forest_roc_curve.png"           : "Tuned_Random_Forest_roc_curve.png",
    "XGBoost_confusion_matrix.png"                : "XGBoost_confusion_matrix.png",
    "XGBoost_roc_curve.png"                       : "XGBoost_roc_curve.png",
}

MANUAL_SCREENSHOTS = [
    ("banner.png",                  "Put the SVG banner here (rename banner.svg → banner.png, or export as PNG)"),
    ("homepage_screenshot.png",     "Open index.html in browser → take a full-page screenshot"),
    ("dashboard_screenshot.png",    "Open dashboard.html in browser → take a full-page screenshot"),
    ("prediction_result.png",       "Enter values in dashboard → screenshot the result section"),
    ("api_test_screenshot.png",     "Hit POST /predict in Postman or browser → screenshot the JSON response"),
    ("folder_structure_screenshot.png", "Screenshot your VS Code Explorer (the file tree)"),
]

# ─── Setup ─────────────────────────────────────────────────────────────────────

def main():
    print("\n🚀  ExoHabitAI — Assets Setup\n" + "─" * 42)

    # 1. Create assets/images/
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅  Created: {ASSETS_DIR.relative_to(PROJECT_ROOT)}")

    # 2. Auto-copy from reports/
    print("\n📊  Copying report images from reports/ → assets/images/\n")
    copied, missing = 0, 0
    for src_name, dest_name in AUTO_COPY.items():
        src  = REPORTS_DIR / src_name
        dest = ASSETS_DIR / dest_name
        if src.exists():
            shutil.copy2(src, dest)
            print(f"   ✅  {dest_name}")
            copied += 1
        else:
            print(f"   ⚠️   NOT FOUND: reports/{src_name}")
            missing += 1

    print(f"\n   {copied} copied, {missing} missing from reports/")

    # 3. Manual checklist
    print("\n📸  Screenshots you need to take manually:\n")
    for filename, instruction in MANUAL_SCREENSHOTS:
        status = "✅ EXISTS" if (ASSETS_DIR / filename).exists() else "❌ MISSING"
        print(f"   {status}  assets/images/{filename}")
        print(f"            → {instruction}\n")

    # 4. Final status
    total_needed = len(AUTO_COPY) + len(MANUAL_SCREENSHOTS)
    present = sum(1 for f in list(AUTO_COPY.values()) + [m[0] for m in MANUAL_SCREENSHOTS]
                  if (ASSETS_DIR / f).exists())
    print("─" * 42)
    print(f"📁  Total assets present : {present} / {total_needed}")

    if present == total_needed:
        print("🌌  All assets ready — README will render perfectly on GitHub!")
    else:
        remaining = total_needed - present
        print(f"⚡  {remaining} asset(s) still needed — see checklist above.")

    print("\n💡  Tip: After adding all images, run:\n")
    print("       git add assets/")
    print("       git commit -m 'feat: add README assets'")
    print("       git push\n")

if __name__ == "__main__":
    main()
