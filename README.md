# Loan-Default-Prediction

Loan default prediction is an important issue for many banks and financial institutions, as significant losses can occur if borrowers default. This repository contains code for a data science challenge on Coursera aimed at predicting loan defaults using a real-world dataset. XGBoost is chosen for training the classifiers. 





## Evaluation metric
In the context of predicting loan defaults, traditional metrics like accuracy can be misleading because they may primarily reflect the majority class's prevalence rather than the model's ability to accurately identify defaults. The ROC AUC (Receiver Operating Characteristic Area Under the Curve) score is a more appropriate metric as it evaluates the model's performance across all classification thresholds, thereby providing a measure of effectiveness that is unaffected by the imbalance in the classes. Also, correctly identifying potential defaulters is crucial because failures to do so can lead to significant financial losses. Thus, the recall(true positive rate) is another important metric that we would like to optimize.  

##
Result:
![An example image](\Data-Science-Coding-Challenge-Loan-Default-Prediction\result.png)



