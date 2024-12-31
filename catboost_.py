import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
import optuna

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
categorical_features_indices = [X.columns.get_loc(col) for col in categorical_feature_names]

# Use Pool objects for efficiency
train_pool = Pool(data=X, label=y, cat_features=categorical_features_indices)
test_pool = Pool(data=X_test, cat_features=categorical_features_indices)

# Bayesian optimization with Optuna for hyperparameter tuning
def objective(trial):
    params = {
        'iterations': 1000,  # Will rely on early stopping
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'early_stopping_rounds': 30,
        'verbose': False,
        'random_seed': 42,
        'auto_class_weights': 'Balanced'
    }

    # Use cross-validation
    cv_results = cv(
        params=params,
        pool=train_pool,
        fold_count=5,
        shuffle=True,
        partition_random_seed=42,
        early_stopping_rounds=30,
        stratified=True,
        verbose=False
    )

    # Get the iteration with the best AUC score
    best_iteration = cv_results['test-AUC-mean'].idxmax() + 1  # Adjust index if needed
    best_score = cv_results['test-AUC-mean'][best_iteration - 1]
    trial.set_user_attr('best_iteration', best_iteration)
    return best_score

study = optuna.create_study(direction='maximize', study_name='catboost_hyperparameter_optimization')
study.optimize(objective, n_trials=50)  # Adjust n_trials to your needs

print("Best hyperparameters: ", study.best_params)
print("Best AUC: ", study.best_value)
print("Best iteration: ", study.best_trial.user_attrs['best_iteration'])

# Retrieve the best number of iterations
best_iteration = study.best_trial.user_attrs['best_iteration']

# Train the final model with the best hyperparameters on the full training data
best_params = study.best_params
best_model = CatBoostClassifier(
    iterations=best_iteration,
    learning_rate=best_params['learning_rate'],
    depth=best_params['depth'],
    l2_leaf_reg=best_params['l2_leaf_reg'],
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=50,  # To see training progress
    random_seed=42,
    auto_class_weights='Balanced'
)

# Train the model
best_model.fit(train_pool)

# Save the trained model
best_model.save_model('best_catboost_model.cbm')

# Predict probabilities on test data using the trained model
predicted_probs = best_model.predict_proba(test_pool)[:, 1]

# prediction DataFrame
prediction_df = pd.DataFrame({
    'LoanID': test_df['LoanID'],
    'predicted_probability': predicted_probs
})

prediction_df.to_csv('predictions_catboost.csv', index=False)