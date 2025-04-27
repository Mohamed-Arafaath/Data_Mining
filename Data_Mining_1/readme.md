# Data Mining and Analysis of Vocal Emotions for Predictive Modeling
![RAVDESS Dataset](RAVDESS-Dataset.png)

## Overview
This repository presents a comprehensive data mining project conducted as part of the Master's Degree in Data Science and Business Informatics at the University of Pisa. The project leverages the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) to explore vocal emotions through advanced data mining techniques, including clustering, classification, pattern mining, and regression. The goal is to uncover patterns, relationships, and predictive models for vocal emotions, contributing to advancements in audio-based emotion recognition and data science applications like human-computer interaction, sentiment analysis, and audio processing.

---

## Problem Statement
The core objective is to analyze vocal emotion data to extract actionable insights and build predictive models for emotion recognition. The project addresses the following key focus areas:

1. **Data Exploration**: Understand the structure, distribution, and quality of the RAVDESS dataset to identify trends and potential issues.
2. **Clustering**: Group similar audio recordings based on audio features to uncover latent patterns in emotional expressions.
3. **Classification**: Predict the vocal channel (speech vs. song) using machine learning algorithms to evaluate their performance.
4. **Pattern Mining**: Extract frequent patterns and association rules to identify relationships between audio features and vocal channels.
5. **Regression**: Model continuous audio features (e.g., intensity) to assess predictive performance and feature importance.

The project aims to provide a robust framework for analyzing emotional speech and song, with applications in automated audio processing and emotion-aware systems.

---

## Dataset
The RAVDESS dataset consists of 2452 audio recordings from 24 professional actors (12 male, 12 female) expressing emotions in a neutral North American accent. The dataset is available [here](https://drive.google.com/drive/folders/1Azcy3wH9dOLdoYisXC3PgyQBDRM42ydY?usp=drive_link) and includes:

- **Descriptive Attributes**: Modality (audio-only), vocal channel (speech or song), emotion (neutral, calm, happy, sad, angry, fearful, disgust, surprised), emotional intensity (normal, strong), statement ("Kids are talking by the door" or "Dogs are sitting by the door"), repetition, actor ID, and sex.
- **Technical Attributes**: Audio properties like channels, sample width, frame rate, frame width, length (ms), frame count, and intensity (dBFS).
- **Statistical Attributes**: Extracted features including zero-crossing rate, Mel-Frequency Cepstral Coefficients (MFCC), spectral centroid (SC), and Short-Time Fourier Transform (STFT) chromagram statistics (mean, std, min, max, kurtosis, skewness).

---

## Project Structure

The repository contains the following files directly under the `Data_Mining_1/` directory:

- **`DM-1 Project File.ipynb`**:
  - The main Jupyter notebook containing the Python code for the project, including data preprocessing, clustering, classification, pattern mining, regression analyses, and visualizations.

- **`DM-1_Project_Report.pdf`**:
  - The comprehensive project report detailing the methodologies, results, discussions, and visualizations used in the analysis of vocal emotions.

- **`Ravdess_Description.docx`**:
  - A document providing a detailed description of the RAVDESS dataset, including its structure, attributes, and characteristics.

- **`RAVDESS-Dataset.png`**:
  - An image file visualizing or representing the RAVDESS dataset, used in the project documentation.

- **`readme.md`**:
  - The main README file providing an overview of the project, problem statement, dataset details, analysis phases, key findings, tools, and usage instructions.

---

## Analysis and Insights
The project is structured into five key analytical phases, each leveraging specific data mining techniques:

### 1. Data Understanding
- **Exploration**: Analyzed 38 attributes, categorizing them into descriptive, technical, and statistical features. Visualized distributions using histograms, boxplots, and violin plots.
- **Key Insights**: Emotions like anger, fear, and happiness exhibit higher vocal intensity, while sadness is less distinguishable. Gender differences in intensity were minimal.
- **Preliminary Analysis**: Identified missing values (e.g., 33.28% in intensity) and potential outliers for further processing.

### 2. Data Quality
- **Missing Values**: Filled vocal channel (7.99%) with mode, intensity (33.28%) with mean based on correlated attributes, and dropped actor (45.92%) due to excessive missing data.
- **Variable Transformation**: Applied binary encoding for categorical variables (e.g., vocal channel, sex) and ordinal encoding for emotions (0–7 scale). Assessed skewness (0.5) and skipped log-transformation.
- **Correlation Analysis**: Removed redundant attributes (e.g., sample width, frame rate) with single values to reduce dimensionality.
- **Outlier Handling**: Managed outliers using visualizations to ensure robust analysis.

### 3. Data Clustering
- **Algorithms**:
  - **K-means**: Clustered 12 quantitative attributes (e.g., vocal channel, length_ms, MFCC_min) with k=3 (silhouette score: 0.24). Influenced by MFCC_min and frame_count.
  - **DBSCAN**: Identified one cluster and 37 noise points (eps=1.25, minPts=3, silhouette score: 0.33). Noise points corresponded to high kurtosis.
  - **Agglomerative Hierarchical**: Used Ward’s method for balanced clusters (silhouette score: 0.18).
- **Best Algorithm**: DBSCAN, due to its higher silhouette score and ability to identify outliers.

### 4. Classification
- **Target**: Predicted vocal channel (speech vs. song) using:
  - **K-Nearest Neighbors (KNN)**: Achieved 93.6% accuracy (k=13, Euclidean distance). Top features: frame_count (78%), emotion (6.9%), sc_mean (6.6%).
  - **Decision Tree**: Reached 92.4% accuracy (max_depth=4, Gini criterion). Reduced false predictions from 10.88% to 7.58%.
  - **Naive Bayes**: Gaussian Naive Bayes achieved 91.3% accuracy with default parameters.
- **Best Classifier**: KNN, with superior accuracy, F1-score (0.933), and low false prediction rate (6.34%).

### 5. Pattern Mining
- **Frequent Patterns**: Discretized eight variables (e.g., frame_count, kurtosis) and extracted patterns with varying MinSup. At MinSup=0.14, all 20 patterns were “speech.”
- **Association Rules**: Extracted 301 rules (MinSup=0.1, MinConf=60%, min 3-item sets). High-lift rules (e.g., lift=1.796) linked low frame_count to “speech.”
- **Classification**: Used rules for vocal channel prediction, achieving 87.3% accuracy, outperformed by Decision Tree.

### 6. Regression
- **Target**: Predicted audio intensity (loudness in dBFS) using:
  - **Linear Regression**: Achieved RMSE of 2.737 (MAE: 1.944, MSE: 7.491).
  - **Lasso Regression**: Recorded RMSE of 3.132 (MAE: 2.352, MSE: 9.807).
  - **Ridge Regression**: Obtained RMSE of 2.716 (MAE: 1.950, MSE: 7.378).
  - **K-Nearest Neighbors (KNN)**: Attained RMSE of 2.604 (MAE: 2.307, MSE: 6.790).
  - **Decision Tree**: Yielded RMSE of 2.910 (MAE: 1.752, MSE: 8.466).
- **Best Regressor**: KNN, with the lowest RMSE (2.604), indicating superior performance in minimizing prediction errors, particularly for large errors.

---

## Key Findings
- **Emotional Trends**: Anger, fear, and happiness have higher vocal intensity; sadness is less distinguishable.
- **Clustering**: DBSCAN effectively identified clusters and outliers, driven by frame_count and MFCC_min.
- **Classification**: KNN excelled in vocal channel prediction, with frame_count as the dominant feature.
- **Pattern Mining**: High-support patterns favored “speech,” reflecting dataset imbalance.
- **Regression**: KNN Regressor provided robust predictions with minimal errors.

---

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, MLxtend
- **Techniques**: Data preprocessing, clustering (K-means, DBSCAN, hierarchical), classification (KNN, Decision Tree, Naive Bayes), pattern mining (FP-growth, association rules), regression (Linear, Lasso, Ridge, KNN, Decision Tree)

---

## How to Use
1. Clone the repository:
   ```bash
   git clone (https://github.com/Mohamed-Arafaath/Data_Mining/tree/main/Data_Mining_1)
   ```
2. Navigate to the `code/` folder and run the main script: DM-1 Project File.ipynb
3. Access the dataset via the [Google Drive](https://drive.google.com/drive/folders/1Azcy3wH9dOLdoYisXC3PgyQBDRM42ydY?usp=drive_link).
4. Explore visualizations and the project report in the `visualizations/` and `report/` folders.
---

## 📧 Contact

For any questions or feedback, feel free to reach out:

- **Author**: [Mohamed Arafaath](https://www.linkedin.com/in/mohamed-arafaath/)
- **Email**: mohamed_arafaath@outlook.com
