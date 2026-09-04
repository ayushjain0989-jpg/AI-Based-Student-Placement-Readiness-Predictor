import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "dataset/Sample.csv"

TARGET = "Placement(Y/N)?"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("AI-BASED STUDENT PLACEMENT READINESS PREDICTOR")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print("Rows   :", df.shape[0])
print("Columns:", df.shape[1])


print("\nDataset columns:")

for column in df.columns:

    print(" -", column)


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [

    "Gender",

    "10th board",

    "10th marks",

    "12th board",

    "12th marks",

    "Stream",

    "Cgpa",

    "Internships(Y/N)",

    "Training(Y/N)",

    "Backlog in 5th sem",

    "Innovative Project(Y/N)",

    "Communication level",

    "Technical Course(Y/N)",

    TARGET

]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    raise ValueError(

        "\nMissing required columns:\n"

        + "\n".join(
            f" - {column}"
            for column in missing_columns
        )

    )


print("\nAll required columns are present.")


# ============================================================
# 4. REMOVE IDENTIFYING / UNNECESSARY COLUMNS
# ============================================================

columns_to_remove = [

    "Name",

    "Email"

]


df = df.drop(

    columns=columns_to_remove,

    errors="ignore"

)


print("\nName and Email removed from ML features.")


# ============================================================
# 5. CLEAN TARGET VARIABLE
# ============================================================

print("\nTarget variable:", TARGET)


df[TARGET] = (

    df[TARGET]

    .astype(str)

    .str.strip()

    .map({

        "Placed": 1,

        "Not Placed": 0

    })

)


# Remove rows with invalid target values

before_rows = len(df)


df = df.dropna(
    subset=[TARGET]
)


after_rows = len(df)


if before_rows != after_rows:

    print(

        f"\nRemoved "
        f"{before_rows - after_rows} "
        f"rows with invalid target values."

    )


# Convert target to integer

df[TARGET] = df[TARGET].astype(int)


print("\nTarget distribution:")

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. FEATURES
# ============================================================

FEATURES = [

    "Gender",

    "10th board",

    "10th marks",

    "12th board",

    "12th marks",

    "Stream",

    "Cgpa",

    "Internships(Y/N)",

    "Training(Y/N)",

    "Backlog in 5th sem",

    "Innovative Project(Y/N)",

    "Communication level",

    "Technical Course(Y/N)"

]


X = df[FEATURES]

y = df[TARGET]


print("\nNumber of ML features:", len(FEATURES))


# ============================================================
# 7. COLUMN TYPES
# ============================================================

categorical_features = [

    "Gender",

    "10th board",

    "12th board",

    "Stream",

    "Internships(Y/N)",

    "Training(Y/N)",

    "Backlog in 5th sem",

    "Innovative Project(Y/N)",

    "Technical Course(Y/N)"

]


numeric_features = [

    "10th marks",

    "12th marks",

    "Cgpa",

    "Communication level"

]


print("\nNumeric features:")

for feature in numeric_features:

    print(" -", feature)


print("\nCategorical features:")

for feature in categorical_features:

    print(" -", feature)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y

)


print("\n" + "=" * 70)

print("DATA SPLIT")

print("=" * 70)

print(

    "Training samples:",
    len(X_train)

)

print(

    "Testing samples :",
    len(X_test)

)


# ============================================================
# 9. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(

    steps=[

        (

            "imputer",

            SimpleImputer(
                strategy="median"
            )

        )

    ]

)


categorical_transformer = Pipeline(

    steps=[

        (

            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )

        ),

        (

            "onehot",

            OneHotEncoder(

                handle_unknown="ignore",

                sparse_output=False

            )

        )

    ]

)


preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            numeric_transformer,

            numeric_features

        ),

        (

            "cat",

            categorical_transformer,

            categorical_features

        )

    ]

)


# ============================================================
# 10. MACHINE LEARNING MODELS
# ============================================================

models = {

    "Random Forest":

        RandomForestClassifier(

            n_estimators=300,

            max_depth=None,

            random_state=RANDOM_STATE,

            class_weight="balanced"

        ),


    "XGBoost":

        XGBClassifier(

            n_estimators=300,

            max_depth=4,

            learning_rate=0.05,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=RANDOM_STATE,

            eval_metric="logloss"

        ),


    "CatBoost":

        CatBoostClassifier(

            iterations=300,

            depth=5,

            learning_rate=0.05,

            loss_function="Logloss",

            verbose=False,

            random_seed=RANDOM_STATE

        )

}


# ============================================================
# 11. TRAIN AND EVALUATE MODELS
# ============================================================

results = {}

trained_models = {}


for model_name, model in models.items():

    print("\n" + "=" * 70)

    print("Training:", model_name)

    print("=" * 70)


    # --------------------------------------------------------
    # Create independent pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(

        steps=[

            (

                "preprocessor",

                preprocessor

            ),

            (

                "model",

                model

            )

        ]

    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline.fit(

        X_train,

        y_train

    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = pipeline.predict(

        X_test

    )


    probabilities = pipeline.predict_proba(

        X_test

    )[:, 1]


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(

        y_test,

        predictions

    )


    precision = precision_score(

        y_test,

        predictions,

        zero_division=0

    )


    recall = recall_score(

        y_test,

        predictions,

        zero_division=0

    )


    f1 = f1_score(

        y_test,

        predictions,

        zero_division=0

    )


    roc_auc = roc_auc_score(

        y_test,

        probabilities

    )


    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results[model_name] = {

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1 Score":
            f1,

        "ROC-AUC":
            roc_auc

    }


    trained_models[model_name] = pipeline


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
).T


print("\n\n" + "=" * 70)

print("MODEL PERFORMANCE COMPARISON")

print("=" * 70)


print(
    results_df.round(4)
)


# ============================================================
# 13. SELECT BEST MODEL
# ============================================================

best_model_name = (

    results_df[

        "F1 Score"

    ].idxmax()

)


best_model_test = (

    trained_models[
        best_model_name
    ]

)


print("\n" + "=" * 70)

print("BEST MODEL")

print("=" * 70)


print(
    "Selected model:",
    best_model_name
)


print(

    "Selection criterion:",
    "Highest F1 Score"

)


print(

    "F1 Score:",

    f"{results_df.loc[best_model_name, 'F1 Score']:.4f}"

)


# ============================================================
# 14. RETRAIN BEST MODEL ON FULL DATASET
# ============================================================

print("\n" + "=" * 70)

print("FINAL MODEL TRAINING")

print("=" * 70)


print(

    "Retraining",

    best_model_name,

    "using the complete dataset..."

)


# Create a fresh model instance

if best_model_name == "Random Forest":

    final_model_algorithm = RandomForestClassifier(

        n_estimators=300,

        max_depth=None,

        random_state=RANDOM_STATE,

        class_weight="balanced"

    )


elif best_model_name == "XGBoost":

    final_model_algorithm = XGBClassifier(

        n_estimators=300,

        max_depth=4,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=RANDOM_STATE,

        eval_metric="logloss"

    )


else:

    final_model_algorithm = CatBoostClassifier(

        iterations=300,

        depth=5,

        learning_rate=0.05,

        loss_function="Logloss",

        verbose=False,

        random_seed=RANDOM_STATE

    )


# Create final pipeline

final_model = Pipeline(

    steps=[

        (

            "preprocessor",

            preprocessor

        ),

        (

            "model",

            final_model_algorithm

        )

    ]

)


# Train on complete dataset

final_model.fit(

    X,

    y

)


print(

    "Final model trained successfully."

)


# ============================================================
# 15. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(

    "models",

    exist_ok=True

)


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

joblib.dump(

    final_model,

    "models/best_model.pkl"

)


print(

    "\nSaved:",
    "models/best_model.pkl"

)


# ============================================================
# 17. SAVE FEATURE INFORMATION
# ============================================================

joblib.dump(

    FEATURES,

    "models/features.pkl"

)


print(

    "Saved:",
    "models/features.pkl"

)


# ============================================================
# 18. SAVE MODEL PERFORMANCE
# ============================================================

joblib.dump(

    results_df,

    "models/model_results.pkl"

)


print(

    "Saved:",
    "models/model_results.pkl"

)


# ============================================================
# 19. TEST SAMPLE
# ============================================================

sample_student = pd.DataFrame([{

    "Gender":
        "Male",

    "10th board":
        "CBSE",

    "10th marks":
        90,

    "12th board":
        "CBSE",

    "12th marks":
        85,

    "Stream":
        "Computer Science and Engineering",

    "Cgpa":
        8.5,

    "Internships(Y/N)":
        "Yes",

    "Training(Y/N)":
        "Yes",

    "Backlog in 5th sem":
        "No",

    "Innovative Project(Y/N)":
        "Yes",

    "Communication level":
        4,

    "Technical Course(Y/N)":
        "Yes"

}])


# ============================================================
# 20. FINAL SAMPLE PREDICTION
# ============================================================

prediction = final_model.predict(

    sample_student

)[0]


probability = (

    final_model.predict_proba(

        sample_student

    )[0][1]

    * 100

)


print("\n" + "=" * 70)

print("TEST PREDICTION")

print("=" * 70)


if prediction == 1:

    print(
        "Prediction: PLACED"
    )

else:

    print(
        "Prediction: NOT PLACED"
    )


print(

    f"Placement Probability: "
    f"{probability:.2f}%"

)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("TRAINING COMPLETED SUCCESSFULLY")

print("=" * 70)


print(

    "Dataset size:",
    len(df)

)


print(

    "Number of features:",
    len(FEATURES)

)


print(

    "Best model:",
    best_model_name

)


print(

    "Best F1 Score:",

    f"{results_df.loc[best_model_name, 'F1 Score'] * 100:.2f}%"

)


print(

    "\nFiles generated:"

)

print(

    " - models/best_model.pkl"

)

print(

    " - models/features.pkl"

)

print(

    " - models/model_results.pkl"

)

print("\nReady for Flask application.")