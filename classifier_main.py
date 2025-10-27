import pandas as pd
import numpy as np
from collections import Counter
import re
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Data loading
companies_fullset = pd.read_csv('ml_insurance_challenge.csv')
taxonomy_fullset = pd.read_csv('insurance_taxonomy.csv')

print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(f"Companies dataset: {companies_fullset.shape}")
print(f"Taxonomy size: {taxonomy_fullset.shape}")
print(f"Number of insurance labels: {len(taxonomy_fullset)}")

# Missing values analysis
print("\nMissing values per column:")
missing_df = pd.DataFrame({
    'column': companies_fullset.columns,
    'missing_count': companies_fullset.isnull().sum().values,
    'missing_pct': (companies_fullset.isnull().sum().values / len(companies_fullset) * 100).round(2)
})
print(missing_df[missing_df['missing_count'] > 0])


def extract_company_name(description):
    if pd.isna(description):
        return "Unknown_company"
    descriere = str(description)
    patterns = [
        r'^(.*?)\s+is\s+',
        r'^(.*?)\s+,\s',
    ]
    for pattern in patterns:
        match = re.match(pattern, descriere, re.IGNORECASE)
        if match:
            company_name = match.group(1).strip()
            return company_name

    words = description.split()[:5]
    return ' '.join(words) if words else "Unknown_company"


companies_fullset['company_name'] = companies_fullset['description'].apply(extract_company_name)

extracted_count = companies_fullset['company_name'].notna().sum()
total_count = len(companies_fullset)
print(f"\nCompany names extracted: {extracted_count}/{total_count} ({extracted_count / total_count * 100:.1f}%)")

# Sector distribution
print("\nSector distribution:")
sector_dist = companies_fullset['sector'].value_counts()
print(sector_dist)


def clean_combine_info(row):
    tags_text = ""
    if pd.notna(row['business_tags']):
        tags_str = str(row['business_tags'])
        try:
            tags_list = ast.literal_eval(tags_str)
            if isinstance(tags_list, list):
                tags_text = ' '.join(tags_list)
        except:
            tags_text = tags_str.replace('[', '').replace(']', '').replace("'", '').replace('"', '').replace(',', '')

    description = str(row['description']) if pd.notna(row['description']) else ' '
    sector = str(row['sector']) if pd.notna(row['sector']) else ' '
    category = str(row['category']) if pd.notna(row['category']) else ' '
    niche = str(row['niche']) if pd.notna(row['niche']) else ' '
    company_name = str(row['company_name']) if pd.notna(row['company_name']) else ' '

    # Repeat sector for emphasis and combine all
    full_info = f"{sector} {sector} {description} {tags_text} {category} {niche} {company_name}"
    full_info = re.sub(r'\s+', ' ', full_info)
    return full_info


companies_fullset['full_info'] = companies_fullset.apply(clean_combine_info, axis=1)

# Text length distribution
#companies_fullset['info_length'] = companies_fullset['full_info'].apply(lambda x: len(str(x).split()))
#print(f"\nText length statistics (words):")
#print(companies_fullset['info_length'].describe())


class InsuranceClassifier:
    def __init__(self, insurance_labels):
        self.insurance_labels = insurance_labels
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("\nEncoding insurance labels...")
        self.label_embeddings = self.model.encode(insurance_labels, show_progress_bar=True)

        # Sector-specific keyword boosting
        self.sector_boost_map = {
            'Services': {
                'professional': 0.15,
                'liability': 0.1,
                'errors': 0.1,
                'cyber': 0.1
            },
            'Manufacturing': {
                'product': 0.15,
                'liability': 0.1,
                'property': 0.1,
                'equipment': 0.1,
                'industrial': 0.15
            },
            'Wholesale': {
                'cargo': 0.15,
                'warehouse': 0.15,
                'product': 0.1,
                'transport': 0.1
            },
            'Retail': {
                'general': 0.1,
                'liability': 0.1,
                'property': 0.1,
                'theft': 0.15,
                'business': 0.1
            },
            'Government': {
                'public': 0.2,
                'official': 0.15,
                'municipal': 0.15,
                'liability': 0.1
            },
            'Non Profit': {
                'directors': 0.15,
                'officers': 0.15,
                'volunteer': 0.15,
                'donation': 0.1
            },
            'Education': {
                'student': 0.15,
                'educator': 0.15,
                'school': 0.15,
                'academic': 0.1
            }
        }

    def classify_company(self, row, top_n=3, base_threshold=0.3):
        info = row['full_info']
        sector = row.get('sector', '')

        text_embedding = self.model.encode([info])
        similarities = cosine_similarity(text_embedding, self.label_embeddings)[0]

        # Apply sector-specific boosting
        if sector in self.sector_boost_map:
            boosts = self.sector_boost_map[sector]
            for i, label in enumerate(self.insurance_labels):
                label_lower = label.lower()
                for keyword, boost_value in boosts.items():
                    if keyword in label_lower:
                        similarities[i] += boost_value
                        break

        # Dynamic threshold based on text length
        info_length = len(info.split())
        if info_length < 20:
            threshold = base_threshold + 0.2
        elif info_length < 50:
            threshold = base_threshold + 0.1
        else:
            threshold = base_threshold

        # Get top candidates
        top_indices = np.argsort(-similarities)[:top_n * 2]
        results = []
        confidences = []

        for idx in top_indices:
            if similarities[idx] > threshold:
                results.append(self.insurance_labels[idx])
                confidences.append(float(similarities[idx]))
                if len(results) >= top_n:
                    break

        # Fallback to best match if no results above threshold
        if not results:
            best_idx = np.argmax(similarities)
            results = [self.insurance_labels[best_idx]]
            confidences = [float(similarities[best_idx])]

        return {
            'labels': results,
            'confidences': confidences,
            'method': f'hybrid_sector_{sector}' if sector else 'hybrid_no_sector',
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'max_similarity': float(np.max(similarities))
        }



print("CLASSIFICATION PROCESS")


insurance_labels = taxonomy_fullset['label'].tolist()
classifier = InsuranceClassifier(insurance_labels)

results_list = []
for idx, row in companies_fullset.iterrows():
    if idx % 1000 == 0:
        print(f"Processing: {idx}/{len(companies_fullset)} companies")
    result = classifier.classify_company(row)
    results_list.append(result)

# Add results to dataframe
companies_fullset['insurance_labels'] = [r['labels'] for r in results_list]
companies_fullset['label_confidences'] = [r['confidences'] for r in results_list]
companies_fullset['classification_method'] = [r['method'] for r in results_list]
companies_fullset['avg_confidence'] = [r['avg_confidence'] for r in results_list]
companies_fullset['max_similarity'] = [r['max_similarity'] for r in results_list]
companies_fullset['num_labels'] = companies_fullset['insurance_labels'].apply(len)


print("CLASSIFICATION RESULTS SUMMARY")


print(f"\nTotal companies classified: {len(companies_fullset)}")
print(f"Average labels per company: {companies_fullset['num_labels'].mean():.2f}")
print(f"Average confidence: {companies_fullset['avg_confidence'].mean():.2%}")
print(f"Median confidence: {companies_fullset['avg_confidence'].median():.2%}")

print("\nConfidence distribution:")
conf_bins = [0, 0.3, 0.4, 0.5, 0.6, 1.0]
conf_labels = ['<30%', '30-40%', '40-50%', '50-60%', '>60%']
companies_fullset['conf_bin'] = pd.cut(companies_fullset['avg_confidence'], bins=conf_bins, labels=conf_labels)
print(companies_fullset['conf_bin'].value_counts().sort_index())

print("\nLabels per company distribution:")
print(companies_fullset['num_labels'].value_counts().sort_index())

# Most common labels
all_labels = [label for labels in companies_fullset['insurance_labels'] for label in labels]
label_counts = Counter(all_labels)
print("\nTop 15 most assigned labels:")
for label, count in label_counts.most_common(15):
    pct = count / len(companies_fullset) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")

# Low confidence cases
low_confidence = companies_fullset[companies_fullset['avg_confidence'] < 0.4]
print(
    f"\nLow confidence cases (<40%): {len(low_confidence)} ({len(low_confidence) / len(companies_fullset) * 100:.1f}%)")

if len(low_confidence) > 0:
    print("\nExamples requiring review:")
    for idx, row in low_confidence.head(5).iterrows():
        print(f"  - {row['company_name'][:50]}")
        print(f"    Sector: {row['sector']}")
        print(f"    Labels: {row['insurance_labels']}")
        print(f"    Confidence: {row['avg_confidence']:.2%}\n")

# Save results
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save main output
output_df = companies_fullset[['company_name', 'description', 'sector', 'category',
                               'niche', 'business_tags', 'insurance_labels',
                               'label_confidences', 'avg_confidence']].copy()

# Convert list columns to string for CSV compatibility
output_df['insurance_labels'] = output_df['insurance_labels'].apply(str)
output_df['label_confidences'] = output_df['label_confidences'].apply(lambda x: str([f"{c:.4f}" for c in x]))

output_df.to_csv('ml_insurance_challenge_classified.csv', index=False)
print("Main results saved to: ml_insurance_challenge_classified.csv")

# Save detailed results for analysis
companies_fullset.to_csv('detailed_classification_results.csv', index=False)
print("Detailed results saved to: detailed_classification_results.csv")

# Export low confidence cases for manual review
low_confidence_export = low_confidence[['company_name', 'description', 'sector',
                                        'insurance_labels', 'avg_confidence']].copy()
low_confidence_export['insurance_labels'] = low_confidence_export['insurance_labels'].apply(str)
low_confidence_export['manual_review'] = ''
low_confidence_export['correct_labels'] = ''
low_confidence_export.to_csv('manual_review_cases.csv', index=False)
print("Low confidence cases exported to: manual_review_cases.csv")

print("\nClassification complete!")