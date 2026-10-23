# Machine Learning Based Analysis of Study Behaviour Patterns for Student Performance Prediction

MSc Data Science Project (7PAM2002), University of Hertfordshire.
Student: Gnana Theja Valleri (23102882). Supervisor: Aoife Simpson. Resubmission, October 2026.

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
| `Student_Performance_Resubmission.ipynb` | Complete executed analysis, 19 sections, explanation above every step |
| `student_performance.csv` | Dataset (Najem, Ziti and Zaoui Seghroucheni, 2025, Zenodo, DOI 10.5281/zenodo.16459132, CC BY 4.0) |
| `Final_Project_Report_Resubmission.docx` | Final project report |
| `figures/` | Every figure and results table written by the notebook, plus licence and provenance screenshots |
| `_build/` | Scripts that generate the notebook and report (reproducibility only) |

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter nbconvert --to notebook --execute --inplace Student_Performance_Resubmission.ipynb
```

About 20 minutes on a laptop (the GridSearchCV stage dominates). All randomness is fixed with
`RANDOM_STATE = 42`. Tested with Python 3.12, pandas 2.2, scikit-learn 1.8, matplotlib 3.10.

## Headline results

| Model (tuned) | Row level test accuracy | Row level weighted F1 | Grouped test accuracy | GroupKFold accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.267 | 0.226 | 0.271 | 0.268 +/- 0.016 |
| Random Forest | 0.900 | 0.900 | 0.244 | 0.239 +/- 0.014 |
| Gradient Boosting | 0.888 | 0.888 | 0.239 | 0.251 +/- 0.015 |

Row level: 86.4% of test rows share their grade record with the training set; tuned RF accuracy is
99.6% on those rows and 28.6% on unseen records.
