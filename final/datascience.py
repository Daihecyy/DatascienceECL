import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from utils import (
    build_features,
    get_outliers,
    get_sessions,
    read_ds,
)

RANDOM_STATE = 42

## ==== MODEL TRAINING ====
### === Data extraction ===
features_train = read_ds("train")
train_outliers = get_outliers(features_train)
cleaned_features_train = features_train[~train_outliers].reset_index(drop=True)
sessions_train = get_sessions(cleaned_features_train)
X_train_core = build_features(sessions_train).reset_index(drop=True)
X_raw_train = pd.concat(
    [
        X_train_core,
        cleaned_features_train[["navigateur"]],
    ],
    axis=1,
)
y_train, target_categories = pd.factorize(cleaned_features_train["util"])


### === Preprocessing ===
ohe_cols = ["navigateur"]
markers = ["actions", "ecrans", "confs", "chaines"]
preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
        *[
            (
                f"vect_{marker}",
                Pipeline(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(ngram_range=(1, 3), max_features=5000),
                        ),
                        (
                            "svd",
                            TruncatedSVD(
                                n_components=50 if marker != "chaines" else 100,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
                marker,
            )
            for marker in markers
        ],
    ],
    remainder="passthrough",
)


### === Model definition ===
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                max_features=0.8,
                max_depth=40,
                criterion="log_loss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

### === Hyperparameters tuning ===
param_distributions = {
    "classifier__n_estimators": [500, 1000, 1500],
    "classifier__criterion": ["gini", "entropy", "log_loss"],
    "classifier__max_depth": [None, 20, 40, 60],
    "classifier__min_samples_split": [2, 4],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2", 0.8],
    **{
        f"preprocessor__vect_{marker}__svd__n_components": [30, 50, 100]
        for marker in markers
    },
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

search.fit(X_raw_train, y_train)
print(f"\nBest CV F1-macro Score: {search.best_score_:.4f}")
print("Best Parameters:")
for param, val in search.best_params_.items():
    print(f"  {param}: {val}")


## ==== PREDICTION ====
### === Data extraction ===
features_test = read_ds("test")
sessions_test = get_sessions(features_test)
X_test_core = build_features(sessions_test).reset_index(drop=True)
X_raw_test = pd.concat(
    [
        X_test_core,
        features_test[["navigateur"]],
    ],
    axis=1,
)

### === Prediction ===
best_model = search.best_estimator_
predictions_encoded = best_model.predict(X_raw_test)
predictions = target_categories[predictions_encoded]

### === CSV writing ===
submission = pd.DataFrame(
    {"RowId": range(1, len(X_raw_test) + 1), "prediction": predictions}
)
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created successfully!")
