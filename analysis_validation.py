import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("DETAILED ANALYSIS AND VALIDATION")


# Load results
df = pd.read_csv('detailed_classification_results.csv')

# Parse list columns
df['insurance_labels'] = df['insurance_labels'].apply(ast.literal_eval)
df['label_confidences'] = df['label_confidences'].apply(ast.literal_eval)

print(f"\nDataset size: {len(df)} companies")

# Get unique labels count
all_labels_unique = set([label for labels in df['insurance_labels'] for label in labels])
print(f"Unique labels used: {len(all_labels_unique)}")

# 1. Confidence Analysis
print("1. CONFIDENCE ANALYSIS")

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.hist(df['avg_confidence'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Average Confidence')
plt.ylabel('Number of Companies')
plt.title('Distribution of Classification Confidence')
plt.axvline(df['avg_confidence'].mean(), color='r', linestyle='--', label=f'Mean: {df["avg_confidence"].mean():.2%}')
plt.axvline(df['avg_confidence'].median(), color='g', linestyle='--', label=f'Median: {df["avg_confidence"].median():.2%}')
plt.legend()

plt.subplot(1, 3, 2)
confidence_by_sector = df.groupby('sector')['avg_confidence'].mean().sort_values()
confidence_by_sector.plot(kind='barh')
plt.xlabel('Average Confidence')
plt.title('Confidence by Sector')
plt.tight_layout()

plt.subplot(1, 3, 3)
df['num_labels'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Number of Labels')
plt.ylabel('Number of Companies')
plt.title('Distribution of Labels per Company')
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: confidence_analysis.png")

# Statistical summary
print("\nConfidence Statistics:")
print(df['avg_confidence'].describe())

print("\nConfidence by sector:")
conf_by_sector = df.groupby('sector')['avg_confidence'].agg(['mean', 'median', 'std', 'count'])
conf_by_sector = conf_by_sector.sort_values('mean', ascending=False)
print(conf_by_sector)

# 2. Label Distribution Analysis
print("2. LABEL DISTRIBUTION ANALYSIS")

all_labels = [label for labels in df['insurance_labels'] for label in labels]
label_counts = Counter(all_labels)

print(f"\nTotal label assignments: {len(all_labels)}")
print(f"Unique labels used: {len(label_counts)}")
print(f"Average labels per company: {len(all_labels)/len(df):.2f}")

plt.figure(figsize=(14, 6))
top_labels = dict(label_counts.most_common(20))
plt.barh(range(len(top_labels)), list(top_labels.values()))
plt.yticks(range(len(top_labels)), list(top_labels.keys()), fontsize=9)
plt.xlabel('Number of Companies')
plt.title('Top 20 Most Assigned Insurance Labels')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: label_distribution.png")

# Label concentration
print(f"\nTop 10 labels cover {sum([v for k,v in label_counts.most_common(10)])/len(all_labels)*100:.1f}% of all assignments")
print(f"Top 20 labels cover {sum([v for k,v in label_counts.most_common(20)])/len(all_labels)*100:.1f}% of all assignments")

# Rare labels
rare_labels = [k for k, v in label_counts.items() if v < 5]
print(f"\nRare labels (assigned <5 times): {len(rare_labels)}")

# 3. Sector-Label Analysis
print("3. SECTOR-LABEL RELATIONSHIPS")

# Most common label per sector
sector_label_analysis = []
for sector in df['sector'].dropna().unique():
    sector_df = df[df['sector'] == sector]
    sector_labels = [label for labels in sector_df['insurance_labels'] for label in labels]
    if sector_labels:
        most_common = Counter(sector_labels).most_common(3)
        sector_label_analysis.append({
            'sector': sector,
            'companies': len(sector_df),
            'top_label': most_common[0][0] if most_common else None,
            'top_label_count': most_common[0][1] if most_common else 0,
            'avg_confidence': sector_df['avg_confidence'].mean()
        })

sector_analysis_df = pd.DataFrame(sector_label_analysis)
sector_analysis_df = sector_analysis_df.sort_values('companies', ascending=False)
print("\nTop labels by sector:")
print(sector_analysis_df)

# 4. Text Length vs Confidence

print("4. TEXT LENGTH IMPACT")

plt.figure(figsize=(10, 6))
plt.scatter(df['info_length'], df['avg_confidence'], alpha=0.3, s=10)
plt.xlabel('Text Length (words)')
plt.ylabel('Average Confidence')
plt.title('Relationship between Text Length and Confidence')

# Add trend line
z = np.polyfit(df['info_length'], df['avg_confidence'], 2)
p = np.poly1d(z)
x_trend = np.linspace(df['info_length'].min(), df['info_length'].max(), 100)
plt.plot(x_trend, p(x_trend), "r-", linewidth=2, label='Trend')
plt.legend()

plt.tight_layout()
plt.savefig('text_length_vs_confidence.png', dpi=300, bbox_inches='tight')
print("Saved: text_length_vs_confidence.png")

# Length bins analysis
df['length_bin'] = pd.cut(df['info_length'], bins=[0, 20, 50, 100, 500], labels=['<20', '20-50', '50-100', '>100'])
print("\nConfidence by text length:")
print(df.groupby('length_bin', observed=False)['avg_confidence'].agg(['mean', 'median', 'count']))

# 5. Edge Cases and Problematic Classifications

print("5. EDGE CASES ANALYSIS")


# Very low confidence
very_low = df[df['avg_confidence'] < 0.3]
print(f"\nVery low confidence (<30%): {len(very_low)} cases ({len(very_low)/len(df)*100:.2f}%)")

# Single label with low confidence
single_low = df[(df['num_labels'] == 1) & (df['avg_confidence'] < 0.4)]
print(f"Single label + low confidence: {len(single_low)} cases ({len(single_low)/len(df)*100:.2f}%)")

# Many labels (potential over-assignment)
many_labels = df[df['num_labels'] >= 4]
print(f"Many labels (>=4): {len(many_labels)} cases ({len(many_labels)/len(df)*100:.2f}%)")

# 6. Create validation sample

print("6. CREATING VALIDATION SAMPLE")


# Stratified sample for manual validation
validation_sample = pd.DataFrame()

# Sample from different confidence bins
for conf_min, conf_max in [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]:
    subset = df[(df['avg_confidence'] >= conf_min) & (df['avg_confidence'] < conf_max)]
    if len(subset) > 0:
        sample_size = min(10, len(subset))
        validation_sample = pd.concat([validation_sample, subset.sample(sample_size, random_state=42)])

# Add edge cases
if len(very_low) > 0:
    validation_sample = pd.concat([validation_sample, very_low.sample(min(10, len(very_low)), random_state=42)])
if len(many_labels) > 0:
    validation_sample = pd.concat([validation_sample, many_labels.sample(min(10, len(many_labels)), random_state=42)])

# Remove duplicates using index instead of drop_duplicates on list columns
validation_sample = validation_sample[~validation_sample.index.duplicated(keep='first')]
validation_sample = validation_sample.reset_index(drop=True)

# Prepare for manual review
validation_export = validation_sample[['company_name', 'description', 'sector', 'category',
                                       'business_tags', 'insurance_labels', 'avg_confidence']].copy()

# Convert lists to strings for CSV compatibility
validation_export['insurance_labels_str'] = validation_export['insurance_labels'].apply(str)
validation_export['business_tags_str'] = validation_export['business_tags'].apply(str)

# Drop original list columns and rename string versions
validation_export = validation_export.drop(['insurance_labels', 'business_tags'], axis=1)
validation_export = validation_export.rename(columns={
    'insurance_labels_str': 'insurance_labels',
    'business_tags_str': 'business_tags'
})

# Add columns for manual review
validation_export['manual_assessment'] = ''
validation_export['correct_labels'] = ''
validation_export['notes'] = ''

validation_export.to_csv('validation_sample_for_review.csv', index=False)
print(f"\nValidation sample created: {len(validation_sample)} companies")
print("Saved to: validation_sample_for_review.csv")
print("\nPlease review and fill in:")
print("  - 'manual_assessment': correct / incorrect / partial")
print("  - 'correct_labels': if incorrect, what should the labels be")
print("  - 'notes': any observations")

# 7. Summary Statistics

print("7. SUMMARY STATISTICS")


summary_stats = {
    'Total Companies': len(df),
    'Avg Labels per Company': f"{df['num_labels'].mean():.2f}",
    'Avg Confidence': f"{df['avg_confidence'].mean():.2%}",
    'Median Confidence': f"{df['avg_confidence'].median():.2%}",
    'Low Confidence (<40%)': f"{len(df[df['avg_confidence'] < 0.4])} ({len(df[df['avg_confidence'] < 0.4])/len(df)*100:.1f}%)",
    'High Confidence (>60%)': f"{len(df[df['avg_confidence'] > 0.6])} ({len(df[df['avg_confidence'] > 0.6])/len(df)*100:.1f}%)",
    'Unique Labels Used': len(label_counts),
    'Most Common Label': label_counts.most_common(1)[0][0] if label_counts else 'N/A',
}

for key, value in summary_stats.items():
    print(f"{key}: {value}")


