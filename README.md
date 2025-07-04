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
