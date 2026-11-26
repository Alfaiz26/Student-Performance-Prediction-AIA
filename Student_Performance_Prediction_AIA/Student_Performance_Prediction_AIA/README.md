# Student Performance Prediction using Machine Learning (AIA Project)

## Project objective
This project analyzes student-related factors and predicts the numerical exam score (`Exam_Score`) using machine learning.

## Dataset
Kaggle: **Student Performance Factors** by Lai Nguyen  
Dataset page: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

Download `StudentPerformanceFactors.csv` from Kaggle and place it in the `data/` folder.

## Machine learning task
- **Type:** Regression
- **Target:** `Exam_Score`
- **Main model:** Random Forest Regressor
- **Baseline model:** Linear Regression

Random Forest is suitable because the dataset includes numeric and categorical features and may contain nonlinear relationships. It can model interactions and does not require features to have a linear relationship with the target. Linear Regression is included as a simple comparison baseline.

## Project workflow
1. Load and inspect dataset
2. Check missing values and duplicates
3. Remove duplicate records and invalid target values
4. Impute missing numeric values using median and categorical values using mode
5. Encode categorical features using One-Hot Encoding
6. Analyze numeric-feature correlation and save a heatmap
7. Split data into 80% training and 20% testing
8. Train Random Forest and Linear Regression
9. Evaluate using MAE, RMSE and R²
10. Save charts, cleaned data and evaluation results

Preprocessing is placed inside scikit-learn pipelines so imputation and encoding are fitted only on training data, reducing data leakage.

## Folder structure
```text
Student_Performance_Prediction_AIA/
├── data/
│   └── StudentPerformanceFactors.csv   # download from Kaggle
├── outputs/                            # generated when program runs
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to run
1. Install Python 3.10 or newer.
2. Download the dataset and put the CSV in `data/`.
3. Open terminal in this project folder.
4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Run:
   ```bash
   python main.py
   ```

## Output files
The program creates:
- `outputs/cleaned_dataset.csv`
- `outputs/numerical_correlation.csv`
- `outputs/correlation_heatmap.png`
- `outputs/model_evaluation.csv`
- `outputs/actual_vs_predicted.png`
- `outputs/feature_importance.csv`
- `outputs/feature_importance.png`

## Evaluation metrics
- **MAE:** average absolute prediction error; lower is better.
- **RMSE:** penalizes larger errors more; lower is better.
- **R²:** proportion of target variance explained by the model; closer to 1 is generally better.

## Limitations
This is an educational prediction project. Dataset results may not generalize to every school or student population. Feature importance and correlation do not prove cause-and-effect.
