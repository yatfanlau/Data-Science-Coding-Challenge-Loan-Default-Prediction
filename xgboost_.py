import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold

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

# Combine training and test data for consistent encoding
combined_X = pd.concat([X, X_test], axis=0, ignore_index=True)

# Encode categorical features using OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
combined_X[categorical_feature_names] = ordinal_encoder.fit_transform(combined_X[categorical_feature_names])

# Split back into X and X_test
X = combined_X.iloc[:len(X), :]
X_test = combined_X.iloc[len(X):, :]

# Calculate the scale_pos_weight for class imbalance
ratio = float(np.sum(y == 0)) / np.sum(y == 1)

# Using Bayesian optimization with Optuna for hyperparameter tuning
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'tree_method': 'hist',  # 'hist' for faster computation
        'eta': trial.suggest_loguniform('eta', 0.01, 0.3),  # learning_rate
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1e-1),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10),  # L2 regularization
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10),  # L1 regularization
        'scale_pos_weight': ratio,  # For handling class imbalance
        'random_state': 42,
    }

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X, label=y)

    # Use cross-validation 
    cv_results = xgb.cv(
        params=param,
        dtrain=dtrain,
        num_boost_round=1000,
        nfold=5,
        stratified=True,
        metrics='auc',
        early_stopping_rounds=30,
        seed=42,
        verbose_eval=False
    )

    # Extract the best score and iteration
    best_score = cv_results['test-auc-mean'].max()
    best_iteration = cv_results['test-auc-mean'].idxmax() + 1  # Adjust index since it starts from 0

    trial.set_user_attr('best_iteration', best_iteration)

    return best_score

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize', study_name='xgboost_hyperparameter_optimization')
study.optimize(objective, n_trials=50)  # Adjust n_trials to your needs

print("Best hyperparameters: ", study.best_params)
print("Best AUC: ", study.best_value)
print("Best iteration: ", study.best_trial.user_attrs['best_iteration'])

# Retrieve the best number of boosting rounds
best_iteration = study.best_trial.user_attrs['best_iteration']

# best hyperparameters
best_params = study.best_params

# Update parameters with values not tuned in Optuna
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'tree_method': 'hist',  # 'hist' for faster computation
    'scale_pos_weight': ratio,
    'random_state': 42,
})

# Create DMatrix for training and testing
dtrain = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X_test)

# Train the final model with the best hyperparameters on the full training data
bst = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=best_iteration,
    evals=[(dtrain, 'train')],
    verbose_eval=50
)

bst.save_model('best_xgboost_model.json')

# Predict probabilities on test data using the trained model
predicted_probs = bst.predict(dtest)

# Create the prediction DataFrame
prediction_df = pd.DataFrame({
    'LoanID': test_df['LoanID'],
    'predicted_probability': predicted_probs
})

prediction_df.to_csv('predictions_xgboost.csv', index=False)