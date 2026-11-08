import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from utils import (
    build_features,
    clean_action,
    get_outliers,
    get_sessions,
    normalize_name,
    read_ds,
)

RANDOM_STATE = 42

features_train = read_ds("train")
features_test = read_ds("test")

train_outliers = get_outliers(features_train)
cleaned_features_train = features_train[~train_outliers].reset_index(drop=True)

sessions_train = get_sessions(cleaned_features_train)
sessions_test = get_sessions(features_test)

X_train_core = build_features(sessions_train).reset_index(drop=True)
X_test_core = build_features(sessions_test).reset_index(drop=True)

X_raw_train = pd.concat(
    [
        X_train_core,
        cleaned_features_train[["navigateur"]],
    ],
    axis=1,
)
X_raw_test = pd.concat(
    [
        X_test_core,
        features_test[["navigateur"]],
    ],
    axis=1,
)
y, target_categories = pd.factorize(cleaned_features_train["util"])

ohe_cols = ["navigateur"]
ordinal_cols = [
    "ecran_top",
    "conf_top",
    "chaine_top",
]
preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ordinal_cols,
        ),
        # (
        #     "vect",
        #     TfidfVectorizer(
        #         token_pattern=r"(?u)\b\w+\s*->\s*\w+\b",
        #     ),
        #     "actions_bigrams_str",  # <--- Changed from ["ecran_bigrams_str"] to a string
        # ),
    ],
    remainder="passthrough",
)
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ]
)

param_distributions = {
    "classifier__n_estimators": [500, 800, 1000],
    "classifier__criterion": ["gini", "entropy", "log_loss"],
    "classifier__max_depth": [None, 20, 40, 60],
    "classifier__min_samples_split": [2, 4],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2", 0.8],
}
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    cv=cv,
    scoring="f1_macro",
    random_state=RANDOM_STATE,
    verbose=1,
)

print("Starting randomized hyperparameter search with cross-validation...")
search.fit(X_raw_train, y)

print(f"\nBest CV F1-macro Score: {search.best_score_:.4f}")
print("Best Parameters:")
for param, val in search.best_params_.items():
    print(f"  {param}: {val}")

best_model = search.best_estimator_

classifier = best_model.named_steps["classifier"]
preprocessor = best_model.named_steps["preprocessor"]
feature_names = preprocessor.get_feature_names_out()
importances = classifier.feature_importances_
importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
).sort_values(by="importance", ascending=False)
print(importance_df.head(20))


predictions_encoded = best_model.predict(X_raw_test)
predictions = target_categories[predictions_encoded]

submission = pd.DataFrame(
    {"RowId": range(1, len(X_raw_test) + 1), "prediction": predictions}
)
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created successfully!")
