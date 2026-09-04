import joblib
import pandas as pd
import numpy as np
import shap


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "models/best_model.pkl"

model_pipeline = joblib.load(
    MODEL_PATH
)


# =========================================================
# CLEAN FEATURE NAME
# =========================================================

def clean_feature_name(feature_name):

    feature_name = feature_name.replace(
        "num__",
        ""
    )

    feature_name = feature_name.replace(
        "cat__",
        ""
    )


    replacements = {

        "Cgpa":
            "CGPA",

        "10th marks":
            "10th Marks",

        "12th marks":
            "12th Marks",

        "Communication level":
            "Communication Skills",

        "Internships(Y/N)":
            "Internship Experience",

        "Training(Y/N)":
            "Technical Training",

        "Innovative Project(Y/N)":
            "Innovative Project",

        "Technical Course(Y/N)":
            "Technical Course",

        "Backlog in 5th sem":
            "Academic Backlog",

        "10th board":
            "10th Board",

        "12th board":
            "12th Board",

        "Gender":
            "Gender",

        "Stream":
            "Engineering Stream"

    }


    for old_name, new_name in replacements.items():

        if old_name in feature_name:

            return new_name


    return feature_name


# =========================================================
# STUDENT-FRIENDLY EXPLANATION
# =========================================================

def get_feature_explanation(
    feature,
    value,
    student_data
):

    # -----------------------------------------------------
    # Positive impact
    # -----------------------------------------------------

    if value >= 0:

        explanations = {

            "CGPA":
                "Your CGPA positively influenced your placement readiness prediction.",

            "10th Marks":
                "Your 10th standard academic performance positively influenced the prediction.",

            "12th Marks":
                "Your 12th standard academic performance positively influenced the prediction.",

            "Communication Skills":
                "Your communication level positively influenced your placement readiness.",

            "Internship Experience":
                "Your internship experience contributed positively to your placement readiness.",

            "Technical Training":
                "Your technical training contributed positively to the prediction.",

            "Innovative Project":
                "Having an innovative project strengthened your placement readiness profile.",

            "Technical Course":
                "Your technical course experience positively influenced the prediction.",

            "Academic Backlog":
                "Your academic record contributed positively to the prediction.",

            "10th Board":
                "Your 10th board information contributed positively to the model prediction.",

            "12th Board":
                "Your 12th board information contributed positively to the model prediction.",

            "Engineering Stream":
                "Your engineering stream contributed positively to the model prediction.",

            "Gender":
                "This feature contributed positively to the model prediction."

        }

    # -----------------------------------------------------
    # Negative impact
    # -----------------------------------------------------

    else:

        explanations = {

            "CGPA":
                "Your CGPA reduced the predicted placement readiness. Improving academic performance may help.",

            "10th Marks":
                "Your 10th standard marks reduced the predicted placement readiness.",

            "12th Marks":
                "Your 12th standard marks reduced the predicted placement readiness.",

            "Communication Skills":
                "Your communication level reduced the predicted placement readiness. More communication practice may help.",

            "Internship Experience":
                "The absence or level of internship experience reduced the predicted placement readiness.",

            "Technical Training":
                "Your technical training profile reduced the predicted placement readiness.",

            "Innovative Project":
                "Your project profile reduced the predicted placement readiness. Building a stronger project may help.",

            "Technical Course":
                "Your technical course profile reduced the predicted placement readiness. Consider completing relevant courses.",

            "Academic Backlog":
                "Your academic backlog information reduced the predicted placement readiness.",

            "10th Board":
                "This academic-board feature had a negative influence on the model prediction.",

            "12th Board":
                "This academic-board feature had a negative influence on the model prediction.",

            "Engineering Stream":
                "Your engineering stream had a negative influence on the model prediction.",

            "Gender":
                "This feature had a negative influence on the model prediction."

        }


    return explanations.get(
        feature,
        (
            "This factor positively influenced "
            "your placement readiness prediction."
            if value >= 0
            else
            "This factor negatively influenced "
            "your placement readiness prediction."
        )
    )


# =========================================================
# SHAP EXPLANATION
# =========================================================

def get_shap_explanation(
    input_data
):

    # -----------------------------------------------------
    # GET PIPELINE COMPONENTS
    # -----------------------------------------------------

    preprocessor = (
        model_pipeline
        .named_steps[
            "preprocessor"
        ]
    )

    model = (
        model_pipeline
        .named_steps[
            "model"
        ]
    )


    # -----------------------------------------------------
    # TRANSFORM INPUT
    # -----------------------------------------------------

    transformed_data = (
        preprocessor.transform(
            input_data
        )
    )

    transformed_data = np.asarray(
        transformed_data
    )


    # -----------------------------------------------------
    # FEATURE NAMES
    # -----------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    # -----------------------------------------------------
    # SHAP EXPLAINER
    # -----------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )


    shap_values = explainer.shap_values(
        transformed_data
    )


    # -----------------------------------------------------
    # HANDLE SHAP OUTPUT
    # -----------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        values = np.asarray(
            shap_values[1][0]
        )

    else:

        shap_values = np.asarray(
            shap_values
        )


        if shap_values.ndim == 3:

            values = shap_values[
                0,
                :,
                1
            ]

        else:

            values = shap_values[
                0
            ]


    # -----------------------------------------------------
    # CREATE DATAFRAME
    # -----------------------------------------------------

    explanation = pd.DataFrame({

        "feature":
            feature_names,

        "shap_value":
            values

    })


    # -----------------------------------------------------
    # CLEAN NAMES
    # -----------------------------------------------------

    explanation[
        "feature"
    ] = explanation[
        "feature"
    ].apply(
        clean_feature_name
    )


    # -----------------------------------------------------
    # COMBINE ONE-HOT FEATURES
    # -----------------------------------------------------

    explanation = (

        explanation
        .groupby(
            "feature",
            as_index=False
        )[
            "shap_value"
        ]
        .sum()

    )


    # -----------------------------------------------------
    # IMPORTANCE
    # -----------------------------------------------------

    explanation[
        "importance"
    ] = explanation[
        "shap_value"
    ].abs()


    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    explanation = (

        explanation
        .sort_values(
            "importance",
            ascending=False
        )

    )


    return explanation