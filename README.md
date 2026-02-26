# LAB 1
# German Credit Data Analysis

## Overview
This project performs exploratory data analysis (EDA) and feature engineering on the German Credit dataset to understand factors affecting creditworthiness. The goal is to develop a custom rule‑based scoring model without relying on machine learning libraries, achieving at least 60% prediction accuracy on a held‑out test set.

## Objectives
- Load and inspect the dataset.
- Identify and handle missing values.
- Classify features into quantitative, binary, ordinal, and categorical types.
- Perform statistical grouping and aggregation (e.g., average loan amount by purpose).
- Create informative visualizations (histograms, scatter plots, violin plots, 3D plots) using Matplotlib and Seaborn.
- Transform all features into numerical representations (binary/ordinal encoding, scaling).
- Engineer new features based on observed patterns.
- Design and implement a rule‑based scoring function to predict credit risk.
- Evaluate the model on a test set and aim for accuracy ≥ 0.6.

## Technologies Used
- **Python 3.13**
- **Pandas** – data manipulation and analysis
- **NumPy** – numerical operations
- **Matplotlib** / **Seaborn** – data visualization
- **scikit‑learn** – train/test splitting and evaluation

## Key Tasks Performed
1. **Data Loading & Inspection**  
   Loaded the dataset and examined its structure (first/last rows, shape).

2. **Feature Classification**  
   Identified quantitative, binary, ordinal, and categorical features; verified no missing values.

3. **Statistical Summaries**  
   Used `groupby` to compute average credit amount per purpose.

4. **Exploratory Visualizations**  
   - Trigonometric function plots (training with numpy/matplotlib).  
   - Mirror histograms comparing good/bad credit by account status.  
   - Count plots with Seaborn for savings account categories.  
   - Violin plots for age distribution across savings and credit risk.  
   - Layered histograms for age distributions of good, bad, and all cases.  
   - 2D scatter plot of duration vs. amount, colored by risk.  
   - 3D scatter plot adding age as a third dimension.

5. **Feature Transformation**  
   - Mapped binary and ordinal features to numeric codes (0…n‑1).  
   - Dropped purely categorical columns.  
   - Scaled selected numerical features based on observed thresholds (e.g., duration > 24 months flagged as high risk).

6. **Rule‑Based Scoring Model**  
   - Derived a weighted risk score using domain knowledge and patterns observed during EDA.  
   - Chose a threshold to separate good from bad credit applicants.  
   - Achieved **~0.62 accuracy** on the test set, meeting the 0.6 target.

## Skills Demonstrated
- **Python Programming** (Pandas, NumPy, Matplotlib, Seaborn)
- **Exploratory Data Analysis (EDA)**
- **Data Visualization** (univariate, bivariate, multivariate plots)
- **Feature Engineering** (encoding, scaling, creation of new features)
- **Statistical Thinking** (pattern recognition, hypothesis testing)
- **Rule‑Based Model Development** (manual scoring without ML libraries)

## Results
The custom scoring model reached **62% accuracy** on a 25% test split, confirming that simple heuristics derived from EDA can effectively predict credit risk. The analysis also revealed key predictors such as account status, savings, credit history, and loan duration.

# LAB 2
# Gradient Descent for Linear Regression

Implementation and analysis of gradient descent and stochastic gradient descent algorithms for linear regression from scratch using NumPy.

## Overview

This project explores how different optimization parameters affect the convergence of gradient-based learning algorithms:

- **Gradient Descent (GD)**: Full-batch gradient descent with different learning rates
- **Stochastic Gradient Descent (SGD)**: Mini-batch variations with different batch sizes

## Key Features

- Custom `MSELoss` class implementing MSE loss and gradient calculations
- `gradient_descent` and `stochastic_gradient_descent` functions
- Visualization of optimization trajectories in 2D loss landscapes
- Hyperparameter analysis (learning rates: 0.001-0.012, batch sizes: 5-100)

## Key Learnings

- Learning rate critically affects convergence speed and stability
- Larger batch sizes provide smoother convergence at computational cost
- Optimal parameters balance speed and stability for given data

## Technologies Used

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

# LAB 3
# Decision Trees and Random Forest

## Overview
This lab focuses on implementing decision trees from scratch and applying ensemble methods (bagging, random forest) for classification and regression tasks. The work is divided into three parts:

- **Part 1:** Basic calculations for decision trees – entropy, Gini index, information gain, and leaf predictions.
- **Part 2:** From‑scratch implementation of a decision tree (supports real and categorical features, missing values, feature importance) and its evaluation on California housing, student knowledge, and mushroom datasets.
- **Part 3:** Application of `BaggingClassifier` and `RandomForestClassifier` on the Pima Indians Diabetes dataset, including hyperparameter tuning, cross‑validation, and feature importance analysis.

## Key Tasks and Results

### Part 1 – Theoretical Foundations
- Computed entropy and Gini index for sample splits.
- Derived leaf values for regression (mean of target).

### Part 2 – Custom Decision Tree
- Implemented `find_best_split` (vectorized) using Gini (classification) and variance (regression).
- Built a full `DecisionTree` class with missing‑value handling (most frequent/mean imputation) and feature importance tracking.
- **Datasets used:**
  - **California housing** (regression) – best split on `MedInc` (gain ≈ 0.413).
  - **Student Knowledge** – PEG (previous exam grade) showed the highest information gain (≈0.399).
  - **Mushroom** – categorical dataset; achieved **accuracy 0.9988** on a 50% test split.

### Part 3 – Bagging and Random Forest
- **Dataset:** Pima Indians Diabetes (768 samples, 8 features).
- **Models compared:**
  - Single decision tree (tuned `max_depth`, `min_samples_leaf`) – Accuracy 0.7706, AUC‑ROC 0.7937.
  - Bagging (50 trees) – Accuracy 0.7489, **AUC‑ROC 0.8083**.
  - Random Forest (tuned via 5‑fold CV) – Accuracy 0.7143, AUC‑ROC 0.7974.
- **Feature importance:** Glucose was the most important predictor (≈0.301), followed by BMI and Age.

## Skills Demonstrated
- **Machine Learning:** Decision trees, ensemble methods (bagging, random forest), hyperparameter tuning, cross‑validation.
- **Model Evaluation:** Accuracy, precision, recall, F1‑score, AUC‑ROC.
- **Data Preprocessing:** Missing value imputation, label encoding.
- **Algorithm Implementation:** Custom tree from scratch (split search, Gini/variance criteria, recursion).
- **Libraries:** Python, NumPy, Pandas, Scikit‑learn, Matplotlib.

