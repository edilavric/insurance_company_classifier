import pandas as pd
import numpy as np
from collections import Counter
import ast
from datetime import datetime

print("Generating comprehensive analysis report...")

# Load data
df = pd.read_csv('detailed_classification_results.csv')
df['insurance_labels'] = df['insurance_labels'].apply(ast.literal_eval)
df['label_confidences'] = df['label_confidences'].apply(ast.literal_eval)

# Generate markdown report
report = []
report.append("# Insurance Company Classification - Analysis Report")
report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"\n**Dataset Size:** {len(df)} companies")

report.append("\n## Executive Summary")
report.append(f"\n- Total companies classified: {len(df)}")
report.append(f"- Average confidence: {df['avg_confidence'].mean():.2%}")
report.append(f"- Average labels per company: {df['num_labels'].mean():.2f}")
report.append(f"- High confidence classifications (>60%): {len(df[df['avg_confidence'] > 0.6])} ({len(df[df['avg_confidence'] > 0.6])/len(df)*100:.1f}%)")
report.append(f"- Low confidence classifications (<40%): {len(df[df['avg_confidence'] < 0.4])} ({len(df[df['avg_confidence'] < 0.4])/len(df)*100:.1f}%)")

report.append("\n## Methodology")
report.append("\n### Approach")
report.append("The classification system uses a hybrid approach combining:")
report.append("- **Semantic embeddings** (SentenceTransformer all-MiniLM-L6-v2) for understanding company descriptions")
report.append("- **Sector-specific boosting** to prioritize relevant insurance types based on industry")
report.append("- **Dynamic thresholding** adjusted by text length to handle varying information density")
report.append("- **Multi-label classification** allowing companies to match multiple relevant insurance types")

report.append("\n### Why This Approach")
report.append("- **Embeddings over keywords**: Captures semantic meaning rather than exact word matches")
report.append("- **Sector awareness**: Insurance needs vary significantly by industry")
report.append("- **Multi-label**: Companies often need multiple types of insurance")
report.append("- **Confidence scoring**: Provides transparency for uncertain classifications")

report.append("\n## Results Analysis")

report.append("\n### Confidence Distribution")
conf_stats = df['avg_confidence'].describe()
report.append(f"- Mean: {conf_stats['mean']:.2%}")
report.append(f"- Median: {conf_stats['50%']:.2%}")
report.append(f"- Std Dev: {conf_stats['std']:.3f}")
report.append(f"- Min: {conf_stats['min']:.2%}")
report.append(f"- Max: {conf_stats['max']:.2%}")

report.append("\n### Label Distribution")
all_labels = [label for labels in df['insurance_labels'] for label in labels]
label_counts = Counter(all_labels)
report.append(f"\nTop 10 most assigned labels:")
for label, count in label_counts.most_common(10):
    pct = count / len(df) * 100
    report.append(f"- {label}: {count} companies ({pct:.1f}%)")

report.append("\n### Sector Analysis")
sector_stats = df.groupby('sector').agg({
    'avg_confidence': 'mean',
    'num_labels': 'mean',
    'company_name': 'count'
}).round(3)
sector_stats.columns = ['Avg Confidence', 'Avg Labels', 'Count']
sector_stats = sector_stats.sort_values('Count', ascending=False)

# Format as simple table instead of markdown
report.append("\n| Sector | Avg Confidence | Avg Labels | Count |")
report.append("|--------|----------------|------------|-------|")
for sector, row in sector_stats.iterrows():
    report.append(f"| {sector} | {row['Avg Confidence']:.3f} | {row['Avg Labels']:.1f} | {int(row['Count'])} |")

report.append("\n## Strengths")
report.append("\n### What Works Well")
high_conf_sectors = df[df['avg_confidence'] > 0.6].groupby('sector').size().sort_values(ascending=False)
if len(high_conf_sectors) > 0:
    report.append(f"- High confidence classifications in sectors: {', '.join(high_conf_sectors.head(3).index.tolist())}")
report.append("- Sector-specific boosting improves relevance for industry-specific insurance needs")
report.append("- Multi-label approach captures the reality that most companies need multiple insurance types")
report.append("- Semantic embeddings handle variations in how companies describe themselves")

report.append("\n## Weaknesses and Limitations")
report.append("\n### Known Issues")
low_conf = df[df['avg_confidence'] < 0.4]
report.append(f"- {len(low_conf)} companies ({len(low_conf)/len(df)*100:.1f}%) have low confidence classifications")
report.append("- Short descriptions (<20 words) result in lower confidence due to limited context")
report.append(f"- {len(df[df['sector'].isna()])} companies have missing sector information, reducing classification accuracy")

rare_labels = [k for k, v in label_counts.items() if v < 5]
if rare_labels:
    report.append(f"- {len(rare_labels)} insurance labels are rarely assigned (<5 times), potentially indicating:")
    report.append("  - Very specialized insurance types")
    report.append("  - Threshold too high for these specific labels")
    report.append("  - Dataset lacks companies needing these insurance types")

report.append("\n### Areas for Improvement")
report.append("- Threshold tuning: Current thresholds may be too conservative for some label types")
report.append("- Sector coverage: Some sectors lack specific boosting rules")
report.append("- Label semantics: Some insurance labels may be too similar, causing confusion")
report.append("- Missing data handling: Companies with minimal description get lower quality classifications")

report.append("\n## Scalability")
report.append("\n### Current Performance")
report.append(f"- Processed {len(df)} companies successfully")
report.append("- Embedding model: Efficient for real-time classification")
report.append("- Memory footprint: Manageable with current architecture")

report.append("\n### Scaling to Larger Datasets")
report.append("**For millions of companies:**")
report.append("- Batch encoding with GPU acceleration")
report.append("- Pre-compute label embeddings (done once)")
report.append("- Parallel processing across company batches")
report.append("- Approximate nearest neighbor search for similarity computation")
report.append("- Estimated throughput: 1000-5000 companies/second with optimization")

report.append("\n## Alternative Approaches Considered")
report.append("\n### Not Pursued and Why")
report.append("\n**1. Pure Rule-Based System**")
report.append("- Pros: Fast, interpretable, no model training")
report.append("- Cons: Brittle, requires extensive manual rule creation, poor generalization")
report.append("- Decision: Too limited for diverse company descriptions")

report.append("\n**2. TF-IDF + Classical ML**")
report.append("- Pros: Fast, lightweight")
report.append("- Cons: No semantic understanding, struggles with paraphrasing")
report.append("- Decision: Embeddings provide better semantic matching")

report.append("\n**3. Fine-tuned BERT/GPT**")
report.append("- Pros: Potentially highest accuracy")
report.append("- Cons: Requires labeled training data, computationally expensive, harder to scale")
report.append("- Decision: No ground truth available for fine-tuning, zero-shot approach more practical")

report.append("\n**4. LLM Zero-Shot (GPT-4/Claude)**")
report.append("- Pros: Excellent understanding, minimal setup")
report.append("- Cons: API costs prohibitive at scale, latency issues, rate limits")
report.append("- Decision: Not practical for processing thousands of companies")

report.append("\n**5. Clustering Then Classification**")
report.append("- Pros: Could identify natural groupings")
report.append("- Cons: Adds complexity, clustering quality uncertain without ground truth")
report.append("- Decision: Direct classification simpler and more interpretable")

report.append("\n## Validation Strategy")
report.append("\nWithout ground truth labels, validation relies on:")
report.append("- **Manual review** of stratified sample across confidence levels")
report.append("- **Domain expert review** of sector-specific classifications")
report.append("- **Confidence calibration** analysis to ensure confidence scores are meaningful")
report.append("- **Edge case analysis** to identify systematic errors")
report.append("- **Inter-annotator agreement** if multiple reviewers available")

report.append("\n## Recommendations")
report.append("\n### Immediate Actions")
report.append("1. Complete manual review of validation sample (validation_sample_for_review.csv)")
report.append("2. Adjust thresholds based on precision/recall requirements")
report.append("3. Expand sector boost mappings for underperforming sectors")
report.append("4. Investigate low confidence cases to identify patterns")

report.append("\n### Future Enhancements")
report.append("1. Collect labeled data for supervised fine-tuning")
report.append("2. Implement active learning to prioritize manual review")
report.append("3. Add explanation/reasoning for why specific labels were chosen")
report.append("4. Create feedback loop to continuously improve classifications")
report.append("5. Consider ensemble methods combining multiple approaches")

report.append("\n## Conclusion")
report.append(f"\nThe classification system successfully processes {len(df)} companies with an average confidence of {df['avg_confidence'].mean():.2%}. ")
report.append(f"The hybrid approach balances accuracy with scalability, achieving {len(df[df['avg_confidence'] > 0.6])/len(df)*100:.1f}% high-confidence classifications. ")
report.append("While there is room for improvement, particularly in handling edge cases and rare labels, the solution provides a solid foundation for insurance company classification that can scale to larger datasets with appropriate optimizations.")

# Write report
report_text = '\n'.join(report)
with open('ANALYSIS_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("Report generated: ANALYSIS_REPORT.md")
print("\nKey files generated:")
print("1. ANALYSIS_REPORT.md - Comprehensive analysis")
print("2. confidence_analysis.png - Confidence visualizations")
print("3. label_distribution.png - Label frequency chart")
print("4. text_length_vs_confidence.png - Length-confidence relationship")
print("5. validation_sample_for_review.csv - Sample for manual review")
print("6. detailed_classification_results.csv - Full results with metadata")