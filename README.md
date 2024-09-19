# Loan-Default-Prediction

Loan default prediction is an important issue for many banks and financial institutions, as significant losses can occur if borrowers default. This repository contains code for a data science challenge on Coursera aimed at predicting loan defaults using a real-world dataset. Various classifiers, including logistic regression, random forests, XGBoost, and LightGBM, are applied to the dataset, and the results are compared across these different classifiers






## Class imbalance
Two techniques: under-sampling and over-sampling

## Evaluation metric
In the context of predicting loan defaults, traditional metrics like accuracy can be misleading because they may primarily reflect the majority class's prevalence rather than the model's ability to accurately identify defaults. The ROC AUC (Receiver Operating Characteristic Area Under the Curve) score is a more appropriate metric as it evaluates the model's performance across all classification thresholds, thereby providing a measure of effectiveness that is unaffected by the imbalance in the classes. Also, correctly identifying potential defaulters is crucial because failures to do so can lead to significant financial losses. Thus, the recall(true positive rate) is another important metric that we would like to optimize.  

runfile('C:/Users/yflauag/OneDrive - The Chinese University of Hong Kong/Quant/My_projects/Loan_Default_Prediction/try_version_spyder.py', wdir='C:/Users/yflauag/OneDrive - The Chinese University of Hong Kong/Quant/My_projects/Loan_Default_Prediction')
Fitting 4 folds for each of 1 candidates, totalling 4 fits
Accuracy: 0.6946714356577813
Precision: 0.22827146606674165
Recall: 0.6843525179856115
F1 Score: 0.34234943485351177
ROC AUC Score: 0.7579932144227018
Best hyperpara:  {'alpha': 0.03, 'colsample_bytree': 0.7, 'eta': 0.003, 'gamma': 0, 'lambda': 1.7, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 400, 'subsample': 0.7}

runfile('C:/Users/yflauag/OneDrive - The Chinese University of Hong Kong/Quant/My_projects/Loan_Default_Prediction/try_version_spyder.py', wdir='C:/Users/yflauag/OneDrive - The Chinese University of Hong Kong/Quant/My_projects/Loan_Default_Prediction')
Fitting 5 folds for each of 1 candidates, totalling 5 fits
Accuracy: 0.6946714356577813
Precision: 0.22827146606674165
Recall: 0.6843525179856115
F1 Score: 0.34234943485351177
ROC AUC Score: 0.7579932144227018
Best hyperpara:  {'alpha': 0.03, 'colsample_bytree': 0.7, 'eta': 0.003, 'gamma': 0, 'lambda': 1.7, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 400, 'subsample': 0.7}
