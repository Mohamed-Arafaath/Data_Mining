Data Mining and Analysis of Vocal Emotions for Predictive Modeling

Overview
This project, conducted as part of the Master's Degree in Data Science and Business Informatics at the University of Pisa, focuses on applying advanced data mining techniques to the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). The objective was to uncover patterns, relationships, and predictive models for vocal emotions, contributing to advancements in audio-based emotion recognition and data science. By leveraging clustering, classification, pattern mining, and regression, the project provides a robust framework for analyzing emotional speech and song data, with applications in fields like human-computer interaction, sentiment analysis, and audio processing.
Dataset
The RAVDESS dataset comprises 2452 audio recordings from 24 professional actors (12 male, 12 female) expressing emotions through speech and song in a neutral North American accent. The dataset includes:

Descriptive Attributes: Modality (audio-only), vocal channel (speech or song), emotion (neutral, calm, happy, sad, angry, fearful, disgust, surprised), emotional intensity (normal, strong), statement ("Kids are talking by the door" or "Dogs are sitting by the door"), repetition, actor ID, and sex.
Technical Attributes: Audio properties like channels, sample width, frame rate, frame width, length (ms), frame count, and intensity (dBFS).
Statistical Attributes: Features extracted from audio signals, including zero-crossing rate, Mel-Frequency Cepstral Coefficients (MFCC), spectral centroid (SC), and Short-Time Fourier Transform (STFT) chromagram statistics (mean, std, min, max, kurtosis, skewness).

The dataset is accessible here and is well-suited for exploring vocal emotion recognition due to its structured and validated emotional expressions.
Motivation
The project aimed to harness data mining techniques to extract meaningful insights from vocal emotion data, addressing the growing need for reliable emotion recognition systems in audio processing. By combining exploratory data analysis, clustering, classification, pattern mining, and regression, the project contributes to the advancement of data science methodologies and their application in understanding human emotions through vocal cues.
Methodology
The project was structured into several key phases, each employing specific data mining techniques to analyze the RAVDESS dataset comprehensively:
1. Data Understanding

Conducted an in-depth exploration of the dataset’s 38 attributes, categorizing them into descriptive, technical, and statistical features.
Visualized variable distributions using histograms, boxplots, and violin plots to identify trends, such as higher intensity for emotions like anger and happiness.
Performed preliminary statistical analysis, revealing issues like missing values and potential outliers.

2. Data Quality

Missing Values: Handled missing data in vocal channel (7.99%), intensity (33.28%), and actor (45.92%) attributes. Filled vocal channel with mode, intensity with mean based on correlated attributes, and dropped the actor column due to excessive missing data.
Variable Transformation: Applied binary encoding for categorical variables (e.g., vocal channel, sex) and ordinal encoding for emotions (0–7 scale). Assessed skewness (0.5) and determined log-transformation unnecessary.
Correlation and Variable Elimination: Created a correlation matrix to identify and remove redundant attributes (e.g., sample width, frame rate) with single values, reducing dataset dimensionality.
Outliers: Identified and managed outliers using visualizations to ensure robust analysis.

3. Data Clustering

Applied three clustering algorithms to group similar audio recordings:
K-means: Used 12 quantitative attributes (e.g., vocal channel, length_ms, MFCC_min) with Min-Max scaling. Selected k=3 based on the elbow method and silhouette score (0.24), revealing clusters influenced by MFCC_min and frame_count.
DBSCAN: Identified one cluster and 37 noise points with eps=1.25 and minPts=3, achieving a silhouette score of 0.33. Noise points corresponded to high kurtosis values.
Agglomerative Hierarchical Clustering: Tested four linkage methods (complete, single, average, Ward’s) with Euclidean distance, selecting Ward’s method for balanced clusters (silhouette score: 0.18).


Evaluated clustering performance, selecting DBSCAN as the best algorithm due to its higher silhouette score and ability to identify outliers.

4. Classification

Focused on predicting the vocal channel (speech vs. song) using three algorithms:
K-Nearest Neighbors (KNN): Achieved 93.6% accuracy after hyperparameter tuning (k=13, Euclidean distance), with frame_count, emotion, and sc_mean as top contributors (78%, 6.9%, 6.6%).
Decision Tree: Reached 92.4% accuracy with max_depth=4 and Gini criterion, reducing false predictions from 10.88% to 7.58%. Visualized the tree to interpret splits based on frame_count and emotion.
Naive Bayes: Gaussian Naive Bayes outperformed Bernoulli and Multinomial variants, achieving 91.3% accuracy with default parameters.


KNN was identified as the best classifier due to its superior accuracy, F1-score (0.933), and low false prediction rate (6.34%).

5. Pattern Mining

Discretized eight variables (e.g., vocal channel, frame_count, kurtosis) using quartile-based binning.
Frequent Pattern Extraction: Analyzed frequent, closed, and maximal itemsets with varying MinSup thresholds. At MinSup=0.14, all 20 patterns were associated with “speech,” highlighting dataset imbalance.
Association Rule Mining: Extracted 301 rules with MinSup=0.1, MinConf=60%, and minimum 3-item sets. Rules with high lift (>1.6) indicated strong correlations, e.g., low frame_count predicting “speech” (lift=1.796, confidence=0.882).
Vocal Channel Prediction: Used association rules for classification, achieving 87.3% accuracy on the test set, though outperformed by the Decision Tree classifier.

6. Regression

Applied five regression algorithms (Linear, Lasso, Ridge, KNN, Decision Tree) to predict a continuous target (assumed to be intensity or a derived feature).
Evaluated models using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). KNN Regressor performed best with the lowest RMSE (2.604), indicating robustness against large errors.

Key Findings

Data Insights: Emotions like anger, fear, and happiness exhibit higher vocal intensity, while sadness is less distinguishable. Gender differences in intensity were minimal.
Clustering: DBSCAN effectively identified a single cluster and outliers, with frame_count and MFCC_min as key differentiators.
Classification: KNN excelled in predicting vocal channel, leveraging frame_count and emotion features. Decision Trees provided interpretable rules for distinguishing speech from song.
Pattern Mining: Association rules confirmed strong patterns for “speech” at higher support thresholds, with frame_count and kurtosis as critical predictors.
Regression: KNN Regressor minimized prediction errors, suitable for applications requiring precise continuous predictions.

Tools and Technologies

Programming Language: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, MLxtend (for pattern mining)
Techniques: Data preprocessing, clustering (K-means, DBSCAN, hierarchical), classification (KNN, Decision Tree, Naive Bayes), pattern mining (FP-growth, association rules), regression (Linear, Lasso, Ridge, KNN, Decision Tree)

Contributions

Developed a comprehensive pipeline for analyzing vocal emotion data, from preprocessing to predictive modeling.
Identified key audio features (frame_count, MFCC_min, sc_mean) driving emotion recognition, with potential applications in audio classification systems.
Demonstrated the superiority of DBSCAN for clustering and KNN for classification in handling complex audio datasets.
Provided actionable insights into vocal emotion patterns, enhancing the understanding of emotional expression in speech and song.

Future Work

Incorporate additional audio features (e.g., pitch, tempo) to improve model performance.
Explore deep learning models (e.g., CNNs, RNNs) for enhanced emotion recognition.
Address dataset imbalance by oversampling “song” instances or using synthetic data generation.
Extend the analysis to multimodal data (face-and-voice) for a holistic emotion recognition system.

Conclusion
This project showcases a robust application of data mining techniques to analyze vocal emotions, providing valuable insights for audio-based emotion recognition. By systematically addressing data quality, clustering, classification, pattern mining, and regression, the project contributes to the field of data science and informatics. The findings and methodologies can be leveraged in applications like sentiment analysis, virtual assistants, and automated audio processing systems.
Authors

Mohamed Arafaath Sathik Basha
Vincenzo Rocchi
Cristian Ferrara
Supervised by: Professor Riccardo Guidotti

Academic Context
Master’s Degree in Data Science and Business Informatics, University of Pisa, Academic Year 2022–2023.
Repository Contents

DM1_Report.pdf: Detailed project report (view here).
code/: Python scripts for data preprocessing, clustering, classification, pattern mining, and regression.
visualizations/: Plots and figures (histograms, violin plots, ROC curves, dendrograms, etc.).
dataset/: Link to the RAVDESS dataset (access here).

Installation

Clone the repository:git clone <repository-url>


Install dependencies:pip install -r requirements.txt


Run the main script:python main.py



Usage

Access the dataset via the provided Google Drive link.
Run individual scripts in the code/ directory to reproduce specific analyses (e.g., clustering.py, classification.py).
View visualizations in the visualizations/ directory for insights into data distributions and model performance.

Visualizations


License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or collaboration opportunities, feel free to reach out via GitHub Issues or email at [your-email@example.com].

