# Just-ML-Things-30-Days-of-Code
A personal 30-day journey to sharpen my Machine Learning skills through daily coding

---

# Day 1 (20-06-2025): Intro to ML + Setup
✓ Today I brushed up the basics of Machine Learning — what it is, its types (Supervised, Unsupervised, Reinforcement), and how ML fits into the world of data science and AI.

✓ I also set up my ML environment with Python, Jupyter Notebook, and essential libraries like numpy, pandas, matplotlib, seaborn, and scikit-learn.

## Created three basic files:
✓ - `what_is_ml.md` – Theory and introduction  
✓ - `types_of_ml.md` – Categorization with examples  
✓ - `setup.py` – Environment test + sample code

---

# Day 2 (25-06-2025): Linear Regression
✓ Today I explored **Linear Regression**, one of the simplest and most powerful algorithms in supervised learning.

✓ I learned how to fit a line to data using `scikit-learn`, understand the slope and intercept, and evaluate the model using **R² score**.

## Created two files:
✓ - `linear_regression_theory.md` – Core concept, equations, and use cases  
✓ - `linear_regression_sklearn.py` – Hands-on implementation using a toy dataset (X vs y)

📊 Bonus: Also plotted the regression line to visualize how well the model fits the data.

---

# Day 3 (26-06-2025): Multiple Linear Regression
✓ Today I implemented **Multiple Linear Regression** to predict a target variable based on **two or more input features**.

✓ Learned how to prepare a feature matrix using `pandas`, fit the model with `LinearRegression()` from `scikit-learn`, and interpret the learned coefficients.

✓ Also evaluated the model performance using **R² score** and predicted outputs for new inputs.

## Created two files:
✓ - `multiple_linear_regression_theory.md` – Concept, formula (`y = b₀ + b₁x₁ + b₂x₂ + ...`), and real-life applications  
✓ - `multiple_linear_regression_sklearn.py` – Code using `scikit-learn` with custom input features

📊 Bonus: Plotted Actual vs Predicted values to visually assess model fit.

---

# Day 4 (27-06-2025): Polynomial Regression
✓ Today I implemented **Polynomial Regression** to model non-linear data using polynomial features.

✓ I used `PolynomialFeatures` from `sklearn.preprocessing` to generate higher-degree terms and fit them using `LinearRegression`.

✓ Evaluated model performance using **R² score** and visualized how polynomial fitting improves prediction over simple linear regression.

## Created two files:
✓ - `polynomial_regression_theory.md` – Theory, when to use polynomial models, and key formulas  
✓ - `polynomial_regression_sklearn.py` – Python script with polynomial regression code and visualization

📊 Bonus: Compared linear vs polynomial curve fitting and saw significant improvement for non-linear patterns.

---

# Day 5 (28-06-2025): Logistic Regression
✓ Today I implemented **Logistic Regression**, which is used for binary classification problems.

✓ Explored how the **sigmoid function** converts linear output to probability, and used it to classify outcomes like Admitted (1) or Not (0).

✓ Evaluated the model using **confusion matrix** and **classification report**.

## Created two files:
✓ - `logistic_regression_theory.md` – Explanation of sigmoid, binary classification, and performance metrics  
✓ - `logistic_regression.py` – Code to train and evaluate a logistic regression model using scikit-learn

📊 Bonus: Made predictions on new unseen inputs and analyzed the results.

---

# Day 6 (29-06-2025): K-Nearest Neighbors (KNN)
✓ Today I implemented **KNN classifier**, a simple yet powerful algorithm based on **distance to nearest neighbors**.

✓ Used the classic **Iris dataset**, applied **feature scaling**, and trained the model for `k=3`.

✓ Evaluated performance using **classification report**.

## Created two files:
✓ - `knn_theory.md` – KNN intuition, distance calculation, and pros/cons  
✓ - `knn_classifier.py` – Scikit-learn implementation with Iris dataset and standardization

📊 Bonus: Compared results using different values of k and noted the impact.

---

# Day 7 (30-06-2025): Naive Bayes Classifier
✓ Today I explored **Naive Bayes**, a probabilistic classifier based on **Bayes’ Theorem**.

✓ Used `GaussianNB` for classification on the Iris dataset and learned about when to use Gaussian, Bernoulli, or Multinomial NB.

✓ Focused on speed, simplicity, and surprisingly good accuracy for text-like data.

## Created two files:
✓ - `naive_bayes_theory.md` – Bayes’ formula, assumptions of independence, and use cases  
✓ - `naive_bayes_classifier.py` – Implementation using scikit-learn’s GaussianNB

📊 Bonus: Understood why Naive Bayes is so widely used in spam filtering and NLP tasks.

---

# Day 8 (01-07-2025): Decision Tree Classifier
✓ Learned about tree-based models that split data using decision rules. Implemented using `DecisionTreeClassifier` on the Iris dataset with a visual tree plot.

## Created two files:
✓ - `decision_tree_theory.md`  
✓ - `decision_tree_classifier.py`

📊 Bonus: Visualized the decision boundaries using `plot_tree()` from `sklearn.tree`.

---

# Day 9 (02-07-2025): Random Forest Classifier
✓ Explored the ensemble concept of **Random Forest**, combining multiple trees for more robust predictions. Used 100 trees on the Iris dataset.

## Created two files:
✓ - `random_forest_theory.md`  
✓ - `random_forest_classifier.py`

📊 Bonus: Observed how randomization and aggregation improves accuracy and reduces overfitting.

---

# Day 10 (03-07-2025): Support Vector Machine (SVM)
✓ Implemented **SVM** using `SVC` with RBF kernel. Understood margin maximization and kernel tricks. Feature scaling was necessary.

## Created two files:
✓ - `svm_theory.md`  
✓ - `svm_classifier.py`

📊 Bonus: Tried different kernels like linear and poly to see their effect on classification.

---

# Day 11 (04-07-2025): Principal Component Analysis (PCA)
✓ Today I explored **PCA**, an unsupervised technique to reduce dimensions while retaining maximum variance.

✓ Applied it on the Iris dataset to reduce 4D to 2D and visualized clusters in 2D space.

## Created two files:
✓ - `pca_theory.md`  
✓ - `pca_dimensionality_reduction.py`

📊 Bonus: Checked explained variance ratio to confirm how much information is retained.

---

# Day 12 (05-07-2025): K-Means Clustering
✓ Learned about **K-Means**, an algorithm that groups data into K clusters based on feature similarity.

✓ Used synthetic data to demonstrate clustering and visualized it with centroid locations.

## Created two files:
✓ - `kmeans_theory.md`  
✓ - `kmeans_clustering.py`

📊 Bonus: Understood how the algorithm converges by updating centroids iteratively.
---

# Day 13 (06-07-2025): Hierarchical Clustering
✓ Learned about **Agglomerative Clustering** using dendrograms and linkage methods.

✓ Visualized merges with a dendrogram and clustered synthetic data using Ward linkage.

## Created two files:
✓ - `hierarchical_clustering_theory.md`  
✓ - `hierarchical_clustering.py`

📊 Bonus: Observed how clusters are formed in a bottom-up fashion and explored complete vs average linkage.

---

# Day 14 (07-07-2025): Elbow Method + Silhouette Score
✓ Implemented both **Elbow Method** and **Silhouette Score** to determine the optimal number of clusters for K-Means.

✓ Used synthetic data to show how inertia and silhouette values change with different values of K.

## Created two files:
✓ - `clustering_evaluation_theory.md`  
✓ - `kmeans_evaluation.py`

📊 Bonus: Found K=4 to be optimal for our dataset using both methods.

---

# Day 15 (08-07-2025): Mini Project – Mall Customer Segmentation
✓ Clustered real-world **Mall Customer Data** based on annual income and spending score.

✓ Found optimal clusters (K=5) using the Elbow Method and visualized them with centroids.

✓ Segments were interpretable and could be used for targeted marketing.

## Created two files:
✓ - `mall_segmentation_project.md`  
✓ - `mall_customer_segmentation.py`

📊 Bonus: Identified five distinct customer segments and potential business strategies for each.

# Day 16 (09-07-2025): Credit Card Fraud Detection Using Imbalanced Data Techniques

✓ Today I tackled a real-world problem: detecting **fraudulent credit card transactions** from a highly **imbalanced dataset** (~0.17% fraud).

✓ Learned to handle imbalance using **SMOTE** (Synthetic Minority Oversampling Technique) and built models with **Logistic Regression**, **Random Forest**, and **XGBoost**.

✓ Evaluated models using **Precision, Recall, F1-Score**, and **ROC-AUC** to focus on minimizing false negatives (undetected fraud).

## Created three files:
✓ - `fraud_detection_theory.md` – Explained the challenge of imbalanced data, fraud detection techniques, and SMOTE logic
✓ - `fraud_detection.ipynb` – Data cleaning, SMOTE balancing, model training + ROC curve
✓ - `model_comparison.png` – ROC Curve comparison of all three models

📊 Bonus: XGBoost performed the best with an ROC AUC close to **0.98**, making it ideal for high-recall fraud detection systems.

---

## 📂 Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions with anonymized features (`V1` to `V28`) + `Amount` and `Time`
- Target: `Class` → 0 = Legit, 1 = Fraud

---

## 📈 What I Learned
- Imbalanced datasets need special handling – accuracy alone is misleading.
- SMOTE works well to balance data synthetically.
- ROC-AUC and F1-Score are better metrics than accuracy here.
- Fraud detection models must prioritize **Recall** (catching frauds).

# Day 17 (09-07-2025): Handling Missing Values & Outliers

✓ Today I explored techniques for handling **missing values** using mean/median imputation.

✓ Also learned to detect and remove **outliers** using the IQR method and visualized them using boxplots.

✓ - `missing_outlier_handling_theory.md`  
✓ - `missing_outlier_handling.py`

📊 Bonus: Cleaned a dummy dataset and removed an outlier with age 120!

---

# Day 18 (10-07-2025): Scaling, Normalization & Encoding

✓ Learned how to prepare data for ML by **scaling features** and **encoding categorical variables**.

✓ Applied `StandardScaler`, `LabelEncoder`, and `OneHotEncoder` to transform data correctly.

✓ - `scaling_encoding_theory.md`  
✓ - `scaling_encoding.py`

📊 Bonus: Observed how scaling changes data distribution and why it’s crucial for models like SVM & KNN.

