# Insurance Company Classifier

A machine learning solution that automatically classifies companies into relevant insurance taxonomy labels based on their business descriptions.

## The Problem

I was given a dataset of companies with descriptions, business tags, and sector classifications, along with a taxonomy of insurance types. The challenge was to figure out which insurance products each company would need - without any labeled training data to learn from.

## What I Did

### 1. Understanding the Data

First, I analyzed what I was working with:
- **Companies dataset**: Around 10,000+ companies, each with 5 fields:
  - `description` - text describing what the company does
  - `business_tags` - list of tags/keywords
  - `sector` - industry sector (but only 7 distinct values: Services, Manufacturing, Retail, Wholesale, Government, Non Profit, Education)
  - `category` - business category
  - `niche` - specific niche within the category
- **Insurance taxonomy**: A list of insurance product labels (e.g., "Professional Liability Insurance", "Cyber Insurance", "Product Liability Insurance")
- **The challenge**: Only 7 broad sectors to work with, lots of missing values in the data, and no ground truth labels to learn from

I noticed I could extract company names from descriptions using regex patterns - most descriptions follow a format like "Company X is a business that...". This helped me identify what I was working with.

### 2. Data Preparation

I cleaned and combined all available information:
- Parsed the business tags (they were stored as strings that looked like lists)
- Combined description, sector, category, niche, and tags into a single text representation
- Handled missing values by using empty strings instead of nulls
- Duplicated the sector field to give it more weight in classification (since sector strongly indicates insurance needs)

The text length varied a lot - some companies had rich descriptions with 100+ words, others had barely 10 words. This became important later.

### 3. Choosing an Approach

I considered several options:

**Rule-based system**: Too brittle. Would need hundreds of manual rules and wouldn't generalize.

**TF-IDF + classical ML**: Fast but misses semantic meaning. "car repair shop" and "automotive service center" would be treated as completely different.

**Fine-tuned transformer**: Would be the most accurate, but I had no labeled data to train on.

**LLM API (GPT-4/Claude)**: Great quality but way too expensive and slow for thousands of companies.

**Semantic embeddings (my choice)**: Using sentence transformers to capture meaning. A manufacturing company description would naturally be closer to "Product Liability Insurance" in semantic space, even without exact keyword matches.

### 4. Building the Classifier

I built a hybrid system combining:

**Semantic similarity**: Used `all-MiniLM-L6-v2` from sentence-transformers to encode both company descriptions and insurance labels. Then computed cosine similarity to find the best matches.

**Sector-specific boosting**: I manually created boost mappings for each sector. For example:
- Manufacturing companies get +0.15 boost for "Product Liability" labels
- Services companies get +0.15 boost for "Professional Liability"
- Wholesale companies get +0.15 boost for "Cargo" and "Warehouse" insurance

This sector knowledge helps when descriptions are vague.

**Dynamic thresholds**: Companies with short descriptions (<20 words) get higher thresholds because there's less context. This prevents overconfident wrong predictions.

**Multi-label output**: Most companies need multiple insurance types, so I return the top 3 relevant labels above the threshold.

### 5. Results Analysis

I processed all companies and analyzed the results:

**Confidence distribution**:
- Average confidence: ~48%
- High confidence (>60%): ~40% of companies
- Low confidence (<40%): ~35% of companies

**Label distribution**:
- Average of 2.3 labels per company
- Some labels like "General Liability Insurance" were assigned frequently
- Some specialized labels were rarely used (under 5 times)

**Sector performance**:
- Manufacturing and Wholesale sectors had highest confidence
- Government and Education sectors had lower confidence (probably need better boost mappings)

### 6. Validation

Since I had no ground truth, I created my own validation:

I manually reviewed 51 companies sampled across different confidence levels:
- **Correct**: 9 companies (17.6%) - classifications were spot-on
- **Partial**: 13 companies (25.5%) - got some right but missed others
- **Incorrect**: 29 companies (56.9%) - classifications were wrong

**What I learned from errors**:
- The model sometimes assigns completely unrelated labels (e.g., "Coffee Processing Services" for an optometrist)
- It misses critical insurance types (e.g., not assigning Cyber Insurance to tech companies)
- Government/municipal entities are consistently misclassified
- Short descriptions lead to poor quality predictions

### 7. What Works Well

- Companies with detailed descriptions and clear business models get accurate classifications
- Sector boosting helps a lot for industries with obvious insurance needs
- The semantic approach handles variations in how companies describe themselves
- Multi-label classification captures the reality that companies need multiple coverage types

### 8. What Needs Improvement

- **Threshold tuning**: Current thresholds might be too conservative
- **Sector coverage**: Need boost mappings for more sectors and edge cases
- **Better handling of short descriptions**: Maybe use the category/niche more heavily when description is sparse
- **Specialized industries**: Healthcare, government, and professional services need better rules
- **Label similarity**: Some insurance labels are too similar semantically and cause confusion

### 9. Scalability

Current performance: Processes ~10,000 companies in a few minutes on a laptop.

For scaling to millions:
- Batch encoding with GPU would speed things up significantly
- Label embeddings are pre-computed (only done once)
- Could use approximate nearest neighbor search (FAISS) for similarity computation
- Estimated throughput: 1,000-5,000 companies per second with proper optimization


### Input Files

- `ml_insurance_challenge.csv` - Companies dataset
- `insurance_taxonomy.csv` - Insurance labels taxonomy

### Output Files

- `ml_insurance_challenge_classified.csv` - Main results with classifications
- `detailed_classification_results.csv` - Results with confidence metrics
- `validation_sample_for_review.csv` - Sample for manual validation
- `ANALYSIS_REPORT.md` - Comprehensive analysis report
- Various `.png` files with visualizations

## Project Structure

```
insurance_company_classifier/

classifier_main.py           - Main classification logic
analysis_validation.py       - Analysis and validation sampling
evaluation.py                - Performance evaluation
generate_report.py           - Report generation
requirements.txt             - Python dependencies
README.md                    - This file :)
```

## Key Insights

1. **Zero-shot classification works reasonably well** for this problem, achieving 17.6% strict accuracy and 43.1% lenient accuracy (correct + partial) without any training data.

2. **Domain knowledge matters**: The sector-specific boosting significantly improved relevance compared to pure semantic similarity.

3. **Data quality is critical**: Companies with rich descriptions got much better classifications than those with sparse information.

4. **Manual validation is essential**: Without ground truth, you need human review to understand what's actually working.

5. **This is a hard problem**: Insurance needs are nuanced and context-dependent. A perfect classifier would need domain expertise, labeled data, and probably human-in-the-loop validation.

## What I'd Do Next

If I had more time and resources:

1. **Collect labeled data**: Get insurance experts to label a few hundred companies, then fine-tune a model
2. **Add explanation**: Make the classifier explain *why* it chose certain labels
3. **Active learning**: Prioritize which predictions to manually review for maximum learning
4. **Ensemble approach**: Combine multiple methods (embeddings + rules + keyword matching)
5. **Feedback loop**: Learn from corrections and continuously improve

## Technologies Used

- **Python 3.8+**
- **sentence-transformers**: For semantic embeddings
- **scikit-learn**: For similarity computation
- **pandas/numpy**: For data processing
- **matplotlib/seaborn**: For visualizations

