# GAMMAFEST-2025
GammaFest 2025 challenges participants to build a machine learning model that predicts citation links between scientific papers. Using full texts, metadata, and topic labels, the goal is to help the Elbaf Library recommend relevant research. Models are evaluated using the Matthews Correlation Coefficient (MCC).

# Competition Overview
## Background
For centuries, the Elbaf Library has served as the guardian of the world's knowledge. From ancient wisdom to modern science, everything is meticulously recorded in thousands of papers stored within its walls. The library is watched over by Biblo, a faithful guardian who ensures that knowledge remains accessible to those in need. However, as time has passed, the library’s reference management system has begun to show various weaknesses, hindering access to the vast knowledge it holds.

Now, a group of Holy Knights has arrived, seeking to document their intellectual journeys in the form of scientific papers. They hope to write works that will become a legacy for future generations, but they face a major challenge—how can they find the right references to support their research? They turn to Biblo for help, but Biblo no longer has the strength to assist the Holy Knights in finding, recommending, or connecting relevant papers.

## Problem
The current paper recommendation systems are not effective enough in providing relevant literature suggestions for the Holy Knights. The Elbaf Library needs a system that can help manage and present appropriate references for various research needs. Moreover, mapping relationships between papers remains a complex task, resulting in important references often being overlooked.

Participants in this competition are asked to build a machine learning model to solve this challenge. The model should be able to predict citation relationships between papers. The core issue is to help the Elbaf Library build an optimal reference recommendation system.

To evaluate the models, the Matthews Correlation Coefficient (MCC) will be used as the evaluation metric.

## Dataset Description
The dataset consists of pairs of scientific documents, each equipped with metadata such as titles, abstracts, and publication years. Each row in the dataset represents the possibility that the first document (paper) cites the second document (referenced_paper).

Participants are tasked with building a machine learning model to predict the value of is_referenced, indicating whether a citation relationship actually exists based on the available information.

Files:
- Folder Paper Database: Contains the full texts of documents in .txt format, with filenames corresponding to paper_id.
- papers_metadata.csv: Contains complete metadata for each document.
- train.csv: Contains document pairs with labels indicating whether the paper cites the referenced_paper.
- test.csv: Contains document pairs without labels, to be predicted by participants.
- sample_submission.csv: Sample format for submitting predictions on the test data.

## Metode

### Exploratory Data Analysis (EDA)

The EDA phase was carried out to understand the structure of the data, identify patterns, and discover early insights from the dataset. This process included the following activities:

**1. General Data Information Analysis**
The analysis began by using the `df.info()` function to obtain a summary of the DataFrame, including the number of non-null values in each column and their data types. This helped identify columns with missing values and those needing type conversion.

**2. Initial Data Exploration**
The first 5 rows of the dataset were inspected using `df.head()` to gain an early understanding of the data format and values.

**3. Publication Year Difference Analysis**
A new column 'year_difference' was created by subtracting the publication year of the main document from that of the referenced document. Negative values in 'year_difference' were set to -1 for consistency.

**4. Merged Data Analysis**
A check on combined data information was conducted using `merged_df.info()` to view summary data after merging `train.csv` with `metadata.csv`.

**5. Training Data Analysis**
The original training data was inspected using `train_data.info()` before merging with metadata to understand its basic structure.

### Data Cleaning

The data cleaning phase focused on handling missing values, duplicates, and data inconsistencies. The process included:

**1. Removal of Redundant Columns**
The 'publication_date' column was dropped from the DataFrame since 'publication_year' already exists and the 'year_difference' feature serves as a more informative alternative.

**2. Author Data Cleaning**
The `clean_authors` function was applied to 'clean' the authors column by:
- Replacing NaN with empty strings
- Replacing semicolons (;) with commas (,) for standardization
- Removing non-alphanumeric characters (except commas and spaces)
- Removing double spaces and leading/trailing whitespaces

**3. Text Data Cleaning**
The 'title_paper' and 'title_referenced' columns were cleaned using the clean_text function, which:
- Converts text to lowercase
- Removes non-alphanumeric characters (except commas and spaces)
- Trims excess spaces for formatting consistency

**4. Concept Format Conversion**
The 'concepts_paper' and 'concepts_referenced' columns were converted from semicolon-separated strings into string lists to prepare for encoding.

## Data Preprocessing

The preprocessing stage prepares the data for modeling by conducting feature engineering, encoding, and scaling:

**1. Data Merging**
`train.csv` was merged with `metadata.csv` in two steps:
- First merge for the `paper` data using 'paper_id' as the key
- Second merge for the `referenced_paper` using 'paper_id' as the key
This process enriched the training data with metadata from both papers involved in the reference.

**2. Feature Engineering**
New features were created to improve prediction quality:
- Title Similarity (title_similarity)
  - `HashingVectorizer` was used for memory-efficient vectorization
  - `TfidfVectorizer` was used on a subset for higher-quality embeddings
  - Cosine similarity between main and referenced titles was calculated

- Concept Similarity (concept_sim)
  - Jaccard similarity was calculated between the concept lists of the main and referenced papers
  - A numerical score was assigned to measure topical similarity

- Author Overlap (author_overlap)*
  - Jaccard similarity was calculated between author lists of both papers
  - Last name matching was used to handle name variations

- Citations Feature
  - `citation_ratio`: Citation ratio using `(cited_by_count_paper + 1) / (cited_by_count_referenced + 1)`
  - `citation_diff`: Citation difference using `np.log1p(cited_by_count_paper) - np.log1p(cited_by_count_referenced)`
  - The +1 and `np.log1p` transformations handled zero values and normalized the data

- Age-Adjusted Citation Impact
  - `years_since_pub_ref`: Years since referenced paper publication
  - `citation_rate_ref`: Citation rate per year to measure research impact

**3. Scaling of Numerical Features**
`StandardScaler` was used to normalize numerical features to a uniform scale, enhancing model performance..

**4. Class Imbalance Handling**
Class balancing was performed via:
- Oversampling the minority class (`is_referenced = 1`)
- Undersampling the majority class (`is_referenced = 0`)
- Merging all positive samples with a downsampled set of negative samples

**5. Dataset Splitting**
The dataset was split into training and testing sets using `train_test_split` to ensure objective model evaluation.

### Modeling

The modeling stage used various machine learning algorithms and ensemble techniques:

**1. Individual Models**
Four base models were used in this study:
- `HistGradientBoostingClassifier (HGB)`: A tree-based boosting model efficient for large datasets
- `RandomForestClassifier (RF)`:  An ensemble model using multiple decision trees
- `XGBoost (XGB)`: A popular and optimized gradient boosting implementation
- `Logistic Regression (LR)`: A basic linear model for binary classification

**2. Ensemble Model**
A `VotingClassifier` was implemented with:
- `voting='soft'` to combine probability predictions from all individual models
- Weight optimization for each model using `GridSearchCV` based on the Matthews Correlation Coefficient (MCC)
- Training of individual models during Grid Search to find the optimal weight combination
- Final ensemble model trained using the best weights on the full training set

## Evaluation
Model evaluation was done using various classification metrics to assess prediction performance:

**Evaluation Metrics**
- `Classification Report`: Provides precision, recall, and F1-score for each class, along with accuracy, macro avg, and weighted avg
- `Matthews Correlation Coefficient (MCC)`: A balanced metric for imbalanced datasets, ranging from -1 (total misclassification) to +1 (perfect prediction)

**Model Performance Analysis**
Evaluation results showed:
- Random Forest: MCC = 0.5672
- HistGradientBoosting: MCC = 0.5822 (best performance among individual models)
- Voting Classifier: MCC = 0.5793

**Feature Importance Analysis**
Using `feature_importances_ `from HistGradientBoostingClassifier to identify the most influential features:
1. `primaryGenreName_freq`
2. `downloads_encoded`
3. `appAge`
4. `userRatingCount`
5. `developerCountry_freq`

These features showed significant influence on prediction results and provided insight into the factors influencing citation decisions in scientific research

```
Submissions are evaluated using .csv files with the specified column headers (see the sample submission). The MCC (Matthews Correlation Coefficient) metric will be used to assess the model’s performance. A portion of the test dataset results will form the public leaderboard. The full results will be revealed in the private leaderboard after the competition ends.
```
# Competition Link
https://www.kaggle.com/competitions/gammafest25
