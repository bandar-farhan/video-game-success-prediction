# Video Game Success Prediction

A machine learning project that predicts whether a video game is likely to become commercially successful using pre-release features such as platform, genre, publisher, and release year.

## Project Objective

The goal of this project is to classify video games as:

- Hit
- Not Hit

A game is considered a Hit when its global sales reach at least 1 million units.

## Dataset

The project uses the Video Game Sales dataset with more than 16,000 records.

Main features used:

- Platform
- Year
- Genre
- Publisher

## Machine Learning Models

The following models were implemented and compared:

- Logistic Regression
- Random Forest
- Support Vector Machine with Linear kernel
- Support Vector Machine with RBF kernel
- Support Vector Machine with Polynomial kernel

## Project Workflow

1. Data loading and inspection
2. Missing-value handling
3. Outlier analysis using the IQR method
4. Target-variable creation
5. One-hot encoding
6. Train-test split
7. Feature scaling
8. Model training and evaluation
9. PCA-based model visualization
10. Model comparison

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Key Results

- Random Forest achieved the highest overall accuracy.
- Logistic Regression provided the most stable recall for detecting Hit games.
- The results demonstrate the importance of using recall, precision, and F1-score instead of relying only on accuracy when working with imbalanced data.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Repository Files

- `video_game_success_prediction.py` - Main Python source code
- `vgsales.csv` - Dataset
- `video_game_success_prediction_report.pdf` - Full project report

## How to Run

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Run the project:

```bash
python video_game_success_prediction.py
```

## Authors

This project was completed as a team academic project at the University of Jeddah.
