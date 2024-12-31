import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import joblib
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
train_df = pd.read_csv("train.csv")    # Training data with labels 
test_df = pd.read_csv("test.csv")      # Testing data for making predictions

X = train_df.drop(['Default', 'LoanID'], axis=1)
y = train_df['Default']
X_test = test_df.drop(['LoanID'], axis=1)

# Categorical features
categorical_feature_names = [
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

# Prepare placeholders for Meta features
n_splits = 6
Meta_train = np.zeros((X.shape[0], 3))  # 3 base models
Meta_test = np.zeros((X_test.shape[0], 3))

# Stratified K-Fold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Calculate the scale_pos_weight for class imbalance (used in XGBoost)
ratio = float(np.sum(y == 0)) / np.sum(y == 1)

# best hyperparameters for each model from previous tuning
best_catboost_params = {
    'iterations': 661,  # Replace with best_iteration from CatBoost
    'learning_rate': 0.05804885751362538,  # Replace with your best learning_rate
    'depth': 3,  # Replace with your best depth
    'l2_leaf_reg': 3.130782174196876,  # Replace with your best l2_leaf_reg
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'random_seed': 42,
    'auto_class_weights': 'Balanced',
    'early_stopping_rounds': 30
}

best_xgboost_params = {
    'eta': 0.025827861209301896,  # Replace with your best eta
    'max_depth': 3,  # Replace with your best max_depth
    'min_child_weight': 0.0019473309150178314,  # Replace with your best min_child_weight
    'subsample': 0.5934237482325593,  # Replace with your best subsample
    'colsample_bytree': 0.6504324929704666,  # Replace with your best colsample_bytree
    'gamma': 0.0001213481048730181,  # Replace with your best gamma
    'lambda': 0.02074451609900035,  # Replace with your best lambda
    'alpha': 0.018783933266092283,  # Replace with your best alpha
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'tree_method': 'hist',
    'scale_pos_weight': ratio,
    'random_state': 42
}

best_xgboost_iterations = 836  # Replace with best num_boost_round from XGBoost tuning

best_rf_params = {
    'n_estimators': 900,  # Replace with your best n_estimators
    'max_depth': 15,  # Replace with your best max_depth
    'min_samples_split': 2,  # Replace with your best min_samples_split
    'min_samples_leaf': 10,  # Replace with your best min_samples_leaf
    'max_features': 'sqrt',  # Replace with your best max_features
    'bootstrap': True,  # Replace with your best bootstrap
    'class_weight': 'balanced',  # To handle class imbalance
    'random_state': 42,
    'n_jobs': -1
}

# List to store encoders and preprocessors if needed later
ordinal_encoders = []
onehot_preprocessors = []

# Cross-Validation Loop
for idx, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print(f"Fold {idx+1}")
    
    # Split data
    X_train_fold = X.iloc[train_index].copy()
    y_train_fold = y.iloc[train_index]
    X_valid_fold = X.iloc[valid_index].copy()
    y_valid_fold = y.iloc[valid_index]
    
    # ----- CatBoost -----
    # Get indices of categorical features
    categorical_features_indices = [X.columns.get_loc(col) for col in categorical_feature_names]
    
    # Prepare Pools for CatBoost
    train_pool = Pool(X_train_fold, y_train_fold, cat_features=categorical_features_indices)
    valid_pool = Pool(X_valid_fold, y_valid_fold, cat_features=categorical_features_indices)
    test_pool = Pool(X_test, cat_features=categorical_features_indices)
    
    # Initialize and train CatBoost
    cat_model = CatBoostClassifier(**best_catboost_params)
    cat_model.fit(train_pool, eval_set=valid_pool, verbose=False)
    
    # Predict on validation fold and test set
    Meta_train[valid_index, 0] = cat_model.predict_proba(valid_pool)[:, 1]
    Meta_test[:, 0] += cat_model.predict_proba(test_pool)[:, 1] / n_splits
    
    # ----- XGBoost -----
    # Ordinal encode the categorical variables
    ordinal_encoder = OrdinalEncoder()
    # Fit encoder on training data
    X_train_fold_oe = X_train_fold.copy()
    X_valid_fold_oe = X_valid_fold.copy()
    X_test_oe = X_test.copy()
    X_train_fold_oe[categorical_feature_names] = ordinal_encoder.fit_transform(X_train_fold[categorical_feature_names])
    X_valid_fold_oe[categorical_feature_names] = ordinal_encoder.transform(X_valid_fold[categorical_feature_names])
    X_test_oe[categorical_feature_names] = ordinal_encoder.transform(X_test[categorical_feature_names])
    
    # Optionally, store the encoder if needed later
    ordinal_encoders.append(ordinal_encoder)
    
    # Prepare DMatrix for XGBoost
    dtrain_fold = xgb.DMatrix(X_train_fold_oe, label=y_train_fold)
    dvalid_fold = xgb.DMatrix(X_valid_fold_oe, label=y_valid_fold)
    dtest = xgb.DMatrix(X_test_oe)
    
    # Train XGBoost
    xgb_model = xgb.train(
        params=best_xgboost_params,
        dtrain=dtrain_fold,
        num_boost_round=best_xgboost_iterations,
        evals=[(dvalid_fold, 'validation')],
        early_stopping_rounds=30,
        verbose_eval=False
    )
    
    # Predict on validation fold and test set
    Meta_train[valid_index, 1] = xgb_model.predict(dvalid_fold)
    Meta_test[:, 1] += xgb_model.predict(dtest) / n_splits
    
    # ----- Random Forest -----
    # Define the ColumnTransformer for OneHotEncoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feature_names)
        ],
        remainder='passthrough'  # Keep the remaining columns as is
    )
    
    # Fit the preprocessor on training data
    X_train_fold_rf = preprocessor.fit_transform(X_train_fold)
    X_valid_fold_rf = preprocessor.transform(X_valid_fold)
    X_test_rf = preprocessor.transform(X_test)
    
    onehot_preprocessors.append(preprocessor)
    
    # Initialize and train Random Forest
    rf_model = RandomForestClassifier(**best_rf_params)
    rf_model.fit(X_train_fold_rf, y_train_fold)
    
    # Predict on validation fold and test set
    Meta_train[valid_index, 2] = rf_model.predict_proba(X_valid_fold_rf)[:, 1]
    Meta_test[:, 2] += rf_model.predict_proba(X_test_rf)[:, 1] / n_splits

meta_model = LogisticRegression()
meta_model.fit(Meta_train, y)

joblib.dump(meta_model, 'meta_model.pkl')

# Retrain base models on full training data
print("Retraining base models on full training data...")

# ----- CatBoost -----
# Prepare Pools
full_train_pool = Pool(X, y, cat_features=categorical_features_indices)
full_test_pool = Pool(X_test, cat_features=categorical_features_indices)

# Retrain CatBoost on full data
cat_model_full = CatBoostClassifier(**best_catboost_params)
cat_model_full.fit(full_train_pool, verbose=False)
cat_model_full.save_model('best_catboost_model_full.cbm')

# Predict on test data
Meta_test_full = np.zeros((X_test.shape[0], 3))
Meta_test_full[:, 0] = cat_model_full.predict_proba(full_test_pool)[:, 1]

# ----- XGBoost -----
# Fit OrdinalEncoder on full training data
X_full_oe = X.copy()
X_test_oe = X_test.copy()
ordinal_encoder_full = OrdinalEncoder()
X_full_oe[categorical_feature_names] = ordinal_encoder_full.fit_transform(X[categorical_feature_names])
X_test_oe[categorical_feature_names] = ordinal_encoder_full.transform(X_test[categorical_feature_names])

dtrain_full = xgb.DMatrix(X_full_oe, label=y)
dtest_full = xgb.DMatrix(X_test_oe)

# Retrain XGBoost on full data
xgb_model_full = xgb.train(
    params=best_xgboost_params,
    dtrain=dtrain_full,
    num_boost_round=best_xgboost_iterations,
    evals=[(dtrain_full, 'train')],
    verbose_eval=False
)
xgb_model_full.save_model('best_xgboost_model_full.json')

# Predict on test data
Meta_test_full[:, 1] = xgb_model_full.predict(dtest_full)

# ----- Random Forest -----
# Fit preprocessor on full training data
preprocessor_full = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feature_names)
    ],
    remainder='passthrough'
)
X_full_rf = preprocessor_full.fit_transform(X)
X_test_rf = preprocessor_full.transform(X_test)

# Retrain Random Forest on full data
rf_model_full = RandomForestClassifier(**best_rf_params)
rf_model_full.fit(X_full_rf, y)
joblib.dump({'model': rf_model_full, 'preprocessor': preprocessor_full}, 'best_random_forest_model_full.pkl')

# Predict on test data
Meta_test_full[:, 2] = rf_model_full.predict_proba(X_test_rf)[:, 1]

# Use full Meta_test for final prediction
final_meta_predictions = meta_model.predict_proba(Meta_test_full)[:, 1]

# Create the prediction DataFrame
final_prediction_df = pd.DataFrame({
    'LoanID': test_df['LoanID'],
    'predicted_probability': final_meta_predictions
})

final_prediction_df.to_csv('predictions_combined.csv', index=False)