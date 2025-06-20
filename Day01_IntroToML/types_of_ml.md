# Types of Machine Learning

Machine Learning is broadly categorized into **three main types**, depending on the nature of the data and the learning task.

---

## 1. ✅ Supervised Learning

In this type, the model is trained on **labeled data** (i.e., input and output are both provided).

### Example:
| Input (X) | Output (Y) |
|-----------|------------|
| Hours Studied = 5 | Marks = 80 |
| Hours Studied = 2 | Marks = 50 |

The model learns to map X ➝ Y.

### Use Cases:
- Spam Detection (Email: Spam/Not Spam)
- Credit Scoring (Good/Bad)
- House Price Prediction

### Algorithms:
- Linear Regression
- Logistic Regression
- Decision Trees
- SVM
- KNN

---

## 2. 🔍 Unsupervised Learning

In this type, the model is given **unlabeled data** — it tries to learn patterns or structure from data without output labels.

### Example:
| Input (X) |
|-----------|
| Customer A: Buys milk, bread |
| Customer B: Buys chips, cola |
| Customer C: Buys milk, chips |

The model groups customers with similar buying behavior.

### Use Cases:
- Customer Segmentation
- Market Basket Analysis
- Anomaly Detection

### Algorithms:
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- PCA (Dimensionality Reduction)

---

## 3. 🎮 Reinforcement Learning

In this type, the model learns by **interacting with an environment** and receiving **rewards or penalties** based on its actions.

### Example:
A game-playing AI learns to improve its performance by trial and error — winning gives a reward, losing gives a penalty.

### Use Cases:
- Robotics
- Game AI (e.g., AlphaGo)
- Self-driving Cars
- Dynamic Pricing

### Key Concepts:
- Agent
- Environment
- Action
- Reward

---

## Summary Table

| Type              | Data Type      | Output Available | Goal                        |
|-------------------|----------------|------------------|-----------------------------|
| Supervised        | Labeled         | ✅ Yes           | Predict outcome             |
| Unsupervised      | Unlabeled       | ❌ No            | Find hidden patterns        |
| Reinforcement     | Trial & error   | Depends on reward| Learn through interactions  |

---

> 🎯 Up next: Let’s set up your ML environment and verify everything works in `setup.ipynb`!
