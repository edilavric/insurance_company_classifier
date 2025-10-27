import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import ast


print("EVALUATION ON MANUALLY REVIEWED SAMPLE")


# Load manual review results
try:
    manual_review = pd.read_csv('validation_sample_for_review.csv')
except FileNotFoundError:
    print("Error: validation_sample_for_review.csv not found!")
    print("Please run analysis_validation.py first and complete manual review.")
    exit(1)

# Check if manual review is complete
if manual_review['manual_assessment'].isna().all():
    print("Warning: No manual assessments found in the file!")
    print("Please fill in the 'manual_assessment' column with: correct/incorrect/partial")
    exit(1)

reviewed = manual_review[manual_review['manual_assessment'].notna()]
print(f"\nReviewed samples: {len(reviewed)}/{len(manual_review)}")

if len(reviewed) < 10:
    print("Warning: Very few samples reviewed. Results may not be representative.")

# Calculate metrics
assessments = reviewed['manual_assessment'].str.lower().value_counts()
print("\nManual Assessment Distribution:")
print(assessments)

total_reviewed = len(reviewed)
correct = len(reviewed[reviewed['manual_assessment'].str.lower() == 'correct'])
partial = len(reviewed[reviewed['manual_assessment'].str.lower() == 'partial'])
incorrect = len(reviewed[reviewed['manual_assessment'].str.lower() == 'incorrect'])

accuracy_strict = correct / total_reviewed
accuracy_lenient = (correct + partial) / total_reviewed


print("EVALUATION METRICS")

print(f"Strict Accuracy (only correct): {accuracy_strict:.2%}")
print(f"Lenient Accuracy (correct + partial): {accuracy_lenient:.2%}")

# Confidence correlation with correctness
reviewed['is_correct'] = reviewed['manual_assessment'].str.lower().isin(['correct', 'partial'])
avg_conf_correct = reviewed[reviewed['is_correct']]['avg_confidence'].mean()
avg_conf_incorrect = reviewed[~reviewed['is_correct']]['avg_confidence'].mean()

print(f"\nAverage confidence for correct classifications: {avg_conf_correct:.2%}")
print(f"Average confidence for incorrect classifications: {avg_conf_incorrect:.2%}")
print(f"Confidence gap: {(avg_conf_correct - avg_conf_incorrect):.2%}")

# Analysis by confidence bins
reviewed['conf_bin'] = pd.cut(reviewed['avg_confidence'],
                               bins=[0, 0.3, 0.4, 0.5, 0.6, 1.0],
                               labels=['<30%', '30-40%', '40-50%', '50-60%', '>60%'])

print("\nAccuracy by confidence bin:")
for bin_name in ['<30%', '30-40%', '40-50%', '50-60%', '>60%']:
    bin_data = reviewed[reviewed['conf_bin'] == bin_name]
    if len(bin_data) > 0:
        bin_acc = len(bin_data[bin_data['is_correct']]) / len(bin_data)
        print(f"  {bin_name}: {bin_acc:.2%} (n={len(bin_data)})")

# Sector-wise performance
if 'sector' in reviewed.columns:
    print("\nAccuracy by sector:")
    for sector in reviewed['sector'].dropna().unique():
        sector_data = reviewed[reviewed['sector'] == sector]
        if len(sector_data) >= 3:
            sector_acc = len(sector_data[sector_data['is_correct']]) / len(sector_data)
            print(f"  {sector}: {sector_acc:.2%} (n={len(sector_data)})")

# Error analysis

print("ERROR ANALYSIS")

incorrect_cases = reviewed[~reviewed['is_correct']]
if len(incorrect_cases) > 0:
    print(f"\nTotal incorrect classifications: {len(incorrect_cases)}")
    print("\nExamples of misclassifications:")
    for idx, row in incorrect_cases.head(5).iterrows():
        print(f"\nCompany: {row['company_name']}")
        print(f"Sector: {row.get('sector', 'N/A')}")
        print(f"Predicted: {row['insurance_labels']}")
        print(f"Confidence: {row['avg_confidence']:.2%}")
        if pd.notna(row.get('correct_labels', '')):
            print(f"Correct should be: {row['correct_labels']}")
        if pd.notna(row.get('notes', '')):
            print(f"Notes: {row['notes']}")



print("RECOMMENDATIONS")

if accuracy_strict < 0.6:
    print("- Low accuracy detected. Consider:")
    print("  * Reviewing sector boost mappings")
    print("  * Adjusting threshold values")
    print("  * Adding more domain-specific rules")

if avg_conf_incorrect > 0.5:
    print("- High confidence on incorrect predictions. Consider:")
    print("  * Model may be overconfident")
    print("  * Add diversity penalty for similar labels")

if len(reviewed[reviewed['conf_bin'] == '<30%']) > 0:
    low_conf_acc = len(reviewed[(reviewed['conf_bin'] == '<30%') & reviewed['is_correct']]) / len(reviewed[reviewed['conf_bin'] == '<30%'])
    if low_conf_acc > 0.5:
        print("- Low confidence predictions are somewhat accurate. Consider:")
        print("  * Lowering threshold to capture more labels")

