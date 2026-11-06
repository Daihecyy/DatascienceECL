import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from utils import (
    build_features,
    filter_action,
    get_outliers,
    get_sessions,
    read_ds,
    regex_features,
)

features_train = read_ds("train")
features_test = read_ds("test")

uniques = features_train.iloc[:, 2:].stack().dropna().unique()
filtered_uniques = list({filter_action(un) for un in uniques if not un.startswith("t")})

train_outliers = get_outliers(features_train)
cleaned_features_train = features_train[~train_outliers].reset_index(drop=True)


sessions_train = get_sessions(cleaned_features_train)
sessions_test = get_sessions(features_test)

X_train_core = build_features(sessions_train, filtered_uniques).reset_index(drop=True)
X_test_core = build_features(sessions_test, filtered_uniques).reset_index(drop=True)

X_regex_train = regex_features(sessions_train).reset_index(drop=True)
X_regex_test = regex_features(sessions_test).reset_index(drop=True)

X_raw_train = pd.concat(
    [
        X_train_core,
        cleaned_features_train[["navigateur"]],
        X_regex_train[["ecran_top", "conf_top", "chaine_top"]],
    ],
    axis=1,
)
X_raw_test = pd.concat(
    [
        X_test_core,
        features_test[["navigateur"]],
        X_regex_test[["ecran_top", "conf_top", "chaine_top"]],
    ],
    axis=1,
)

y, target_categories = pd.factorize(cleaned_features_train["util"])

ohe_cols = ["navigateur"]
ordinal_cols = ["ecran_top", "conf_top", "chaine_top"]
preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True), ohe_cols),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ordinal_cols,
        ),
    ],
    remainder="passthrough",
)
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", ExtraTreesClassifier(random_state=42, n_jobs=-1)),
    ]
)

param_distributions = {
    "classifier__n_estimators": [500, 800, 1000],
    "classifier__max_depth": [None, 20, 40, 60],
    "classifier__min_samples_split": [2, 4],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2", 0.8],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    cv=cv,
    scoring="f1_macro",
    random_state=42,
    verbose=1,
)

print("Starting randomized hyperparameter search with cross-validation...")
search.fit(X_raw_train, y)

print(f"\nBest CV F1-macro Score: {search.best_score_:.4f}")
print("Best Parameters:")
for param, val in search.best_params_.items():
    print(f"  {param}: {val}")

best_model = search.best_estimator_

predictions_encoded = best_model.predict(X_raw_test)
predictions = target_categories[predictions_encoded]

submission = pd.DataFrame(
    {"RowId": range(1, len(X_raw_test) + 1), "prediction": predictions}
)
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created successfully!")
