# Machine Learning Based Analysis of Study Behaviour Patterns for Student Performance Prediction

MSc Data Science Project (7PAM2002), University of Hertfordshire.
Student: Gnana Theja Valleri (23102882). Supervisor: Aoife Simpson. Resubmission, 20th November 2026.

## Research question

To what extent can machine learning models predict a student's FinalGrade from behavioural,
engagement, demographic and learning related variables when ExamScore is excluded?

## Short answer

On a row level stratified split a tuned Random Forest reaches 90% accuracy. Tracing the dataset's
construction (an inner join of two Kaggle files, one declared synthetic) shows the 12,469 rows hold only
4,200 distinct grade records, each repeated with different attributes attached. When the split is grouped
by grade record so none appears on both sides, every model falls to about 25%, which is chance.
The behavioural variables do not predict grade in this dataset; the 90% was memorisation.

## Files

| File | Purpose |
|---|---|
| `Student_Performance_Resubmission.ipynb` | Complete executed analysis, 19 sections plus a feature removal test, explanation above every step |
| `merged_dataset.csv` | Dataset downloaded from Zenodo (Najem, Ziti and Zaoui Seghroucheni, 2025, DOI 10.5281/zenodo.16459132, CC BY 4.0). The notebook downloads it from Zenodo if it is missing and checks its MD5 (`f4e2d66446de33c9591292a90e6a7423`) against the Zenodo record before loading |
| `Final_Project_Report_Resubmission.docx` | Final project report |
| `figures/` | Every figure and results table written by the notebook, plus licence and provenance screenshots |
| `models/` | The three tuned models saved with joblib (`random_forest_tuned.joblib`, `gradient_boosting_tuned.joblib`, `logistic_regression_tuned.joblib`) plus `tuned_models.json` with the chosen hyperparameters and test scores. Written by notebook Section 15. Load with `joblib.load("models/random_forest_tuned.joblib")` |

## How to run

```bash
pip install scikit-learn==1.8.0 pandas==2.2.3 numpy==2.2.5 matplotlib jupyter
jupyter nbconvert --to notebook --execute --inplace Student_Performance_Resubmission.ipynb
```

About 20 minutes on a laptop (the GridSearchCV stage dominates). All randomness is fixed with
`RANDOM_STATE = 42`. Tested with Python 3.12, pandas 2.2.3, numpy 2.2.5, scikit-learn 1.8.0, matplotlib 3.10.

Other library versions (for example the defaults on Google Colab) build slightly different trees from
the same seed, so the numbers move in the first decimal place (for example 89.7% instead of 90.0%).
The conclusion does not change. To match the saved outputs exactly on Colab, run
`%pip install -q scikit-learn==1.8.0 pandas==2.2.3 numpy==2.2.5`, restart the session, then Run all.

## Headline results

| Model (tuned) | Row level test accuracy | Row level weighted F1 | Grouped test accuracy | GroupKFold accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.267 | 0.226 | 0.271 | 0.268 +/- 0.016 |
| Random Forest | 0.900 | 0.900 | 0.244 | 0.238 +/- 0.014 |
| Gradient Boosting | 0.888 | 0.888 | 0.238 | 0.251 +/- 0.015 |

Feature removal (tuned RF): dropping the four zero importance Source A features gives 0.902 row level and 0.241 grouped accuracy; the top five features alone give 0.898 and 0.264. Removing features does not change the conclusion.

Row level: 86.4% of test rows share their grade record with the training set; tuned RF accuracy is
99.6% on those rows and 28.6% on unseen records.
