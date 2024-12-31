import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import optuna
import joblib

# Load data
train_df = pd.read_csv("train.csv")    # Training data with labels 
test_df = pd.read_csv("test.csv")      # Testing data for making predictions and submission

X = train_df.drop(['Default', 'LoanID'], axis=1)
y = train_df['Default']
X_test = test_df.drop(['LoanID'], axis=1)

# Categorical features
categorical_feature_names = [
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

# Preprocessing: OneHotEncoding for categorical variables
# Define the ColumnTransformer to do OneHotEncoding on categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feature_names)
    ],
    remainder='passthrough'  # Keep the remaining columns as is
)

# Using Bayesian optimization with Optuna for hyperparameter tuning
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': 'balanced'  # To handle class imbalance
    }
    
    # Create the classifier with suggested hyperparameters
    clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    
    # Create a pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cross-validate
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    mean_score = np.mean(cv_scores)
    
    return mean_score

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize', study_name='rf_hyperparameter_optimization')
study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

print("Best hyperparameters: ", study.best_params)
print("Best AUC: ", study.best_value)

# Train the final model with the best hyperparameters on the full training data

best_params = study.best_params

# Create the final classifier
best_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1, class_weight='balanced')

# Create the pipeline
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_clf)
])

# Train the model on full data
final_pipeline.fit(X, y)

joblib.dump(final_pipeline, 'best_random_forest_model.pkl')

# Predict probabilities on test data using the trained model
predicted_probs = final_pipeline.predict_proba(X_test)[:, 1]

# Create the prediction DataFrame
prediction_df = pd.DataFrame({
    'LoanID': test_df['LoanID'],
    'predicted_probability': predicted_probs
})

prediction_df.to_csv('predictions_random_forest.csv', index=False)