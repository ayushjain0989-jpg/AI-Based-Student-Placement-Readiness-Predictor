from flask import Flask, render_template, request
import pandas as pd
import joblib
import sqlite3
import json
from datetime import datetime

from explain import (
    get_shap_explanation,
    get_feature_explanation
)


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

MODEL_PATH = "models/best_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None


# =========================================================
# DATABASE
# =========================================================

DATABASE = "placement_history.db"


def init_database():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_name TEXT NOT NULL,

            probability REAL NOT NULL,

            status TEXT NOT NULL,

            prediction TEXT NOT NULL,

            assessment_date TEXT NOT NULL,

            student_data TEXT,

            skills TEXT,

            recommendations TEXT,

            shap_data TEXT

        )
    """)

    cursor.execute("""
        PRAGMA table_info(predictions)
    """)

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if "student_data" not in columns:

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN student_data TEXT
        """)

    if "skills" not in columns:

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN skills TEXT
        """)

    if "recommendations" not in columns:

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN recommendations TEXT
        """)

    if "shap_data" not in columns:

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN shap_data TEXT
        """)

    conn.commit()
    conn.close()


# Initialize database
init_database()


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        active_page="home"
    )


# =========================================================
# ASSESSMENT PAGE
# =========================================================

@app.route("/assessment")
def assessment():

    return render_template(
        "assessment.html",
        active_page="assessment"
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

@app.route("/model-performance")
def model_performance():

    try:

        results = joblib.load(
            "models/model_results.pkl"
        )

        print("MODEL RESULTS:")
        print(results)

        model_data = []

        # -------------------------------------------------
        # RESULTS STORED AS DICTIONARY
        # -------------------------------------------------

        if isinstance(results, dict):

            accuracy_data = results.get("Accuracy")
            precision_data = results.get("Precision")
            recall_data = results.get("Recall")

            f1_data = results.get("F1")

            if f1_data is None:
                f1_data = results.get("F1 Score")

            roc_auc_data = results.get("ROC-AUC")

            if roc_auc_data is None:
                roc_auc_data = results.get("ROC AUC")

            # -------------------------------------------------
            # METRIC-ORIENTED FORMAT
            # -------------------------------------------------

            if (
                accuracy_data is not None
                and precision_data is not None
                and recall_data is not None
                and f1_data is not None
                and roc_auc_data is not None
            ):

                if isinstance(accuracy_data, dict):

                    model_names = accuracy_data.keys()

                    for model_name in model_names:

                        accuracy = accuracy_data.get(
                            model_name,
                            0
                        )

                        precision = precision_data.get(
                            model_name,
                            0
                        )

                        recall = recall_data.get(
                            model_name,
                            0
                        )

                        f1 = f1_data.get(
                            model_name,
                            0
                        )

                        roc_auc = roc_auc_data.get(
                            model_name,
                            0
                        )

                        model_data.append({

                            "name": str(model_name),

                            "accuracy": round(
                                float(accuracy) * 100,
                                2
                            ),

                            "precision": round(
                                float(precision) * 100,
                                2
                            ),

                            "recall": round(
                                float(recall) * 100,
                                2
                            ),

                            "f1": round(
                                float(f1) * 100,
                                2
                            ),

                            "roc_auc": round(
                                float(roc_auc) * 100,
                                2
                            )

                        })

            # -------------------------------------------------
            # MODEL-ORIENTED FORMAT
            # -------------------------------------------------

            else:

                for model_name, metrics in results.items():

                    if not isinstance(metrics, dict):
                        continue

                    accuracy = metrics.get(
                        "Accuracy",
                        metrics.get(
                            "accuracy",
                            0
                        )
                    )

                    precision = metrics.get(
                        "Precision",
                        metrics.get(
                            "precision",
                            0
                        )
                    )

                    recall = metrics.get(
                        "Recall",
                        metrics.get(
                            "recall",
                            0
                        )
                    )

                    f1 = metrics.get("F1")

                    if f1 is None:
                        f1 = metrics.get("F1 Score")

                    if f1 is None:
                        f1 = metrics.get(
                            "f1",
                            0
                        )

                    roc_auc = metrics.get(
                        "ROC-AUC"
                    )

                    if roc_auc is None:
                        roc_auc = metrics.get(
                            "ROC AUC"
                        )

                    if roc_auc is None:
                        roc_auc = metrics.get(
                            "roc_auc",
                            0
                        )

                    model_data.append({

                        "name": str(model_name),

                        "accuracy": round(
                            float(accuracy) * 100,
                            2
                        ),

                        "precision": round(
                            float(precision) * 100,
                            2
                        ),

                        "recall": round(
                            float(recall) * 100,
                            2
                        ),

                        "f1": round(
                            float(f1) * 100,
                            2
                        ),

                        "roc_auc": round(
                            float(roc_auc) * 100,
                            2
                        )

                    })

        # -------------------------------------------------
        # RESULTS STORED AS DATAFRAME
        # -------------------------------------------------

        elif isinstance(results, pd.DataFrame):

            for _, row in results.iterrows():

                f1_value = row.get(
                    "F1",
                    row.get(
                        "F1 Score",
                        0
                    )
                )

                roc_auc_value = row.get(
                    "ROC-AUC",
                    row.get(
                        "ROC AUC",
                        0
                    )
                )

                model_name = row.get(
                    "Model",
                    row.get(
                        "model",
                        str(row.name)
                    )
                )

                model_data.append({

                    "name": str(model_name),

                    "accuracy": round(
                        float(
                            row.get(
                                "Accuracy",
                                0
                            )
                        ) * 100,
                        2
                    ),

                    "precision": round(
                        float(
                            row.get(
                                "Precision",
                                0
                            )
                        ) * 100,
                        2
                    ),

                    "recall": round(
                        float(
                            row.get(
                                "Recall",
                                0
                            )
                        ) * 100,
                        2
                    ),

                    "f1": round(
                        float(f1_value) * 100,
                        2
                    ),

                    "roc_auc": round(
                        float(roc_auc_value) * 100,
                        2
                    )

                })

        # -------------------------------------------------
        # CHECK DATA
        # -------------------------------------------------

        if not model_data:

            return render_template(
                "model_performance.html",
                models=[],
                best_model=None,
                active_page="model-performance"
            )

        # -------------------------------------------------
        # BEST MODEL
        # -------------------------------------------------

        best_model = max(
            model_data,
            key=lambda x: x["f1"]
        )

        print("PROCESSED MODEL DATA:")
        print(model_data)

        print("BEST MODEL:")
        print(best_model)

        return render_template(

            "model_performance.html",

            models=model_data,

            best_model=best_model,

            active_page="model-performance"

        )

    except Exception as e:

        print(
            "Model Performance Error:",
            e
        )

        return render_template(

            "model_performance.html",

            models=[],

            best_model=None,

            error="Unable to load model performance data.",

            active_page="model-performance"

        )


# =========================================================
# PREDICTION HISTORY
# =========================================================

@app.route("/history")
def history():

    try:

        conn = sqlite3.connect(
            DATABASE
        )

        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
        """)

        predictions = cursor.fetchall()

        conn.close()

        return render_template(

            "history.html",

            predictions=predictions,

            active_page="history"

        )

    except Exception as e:

        print(
            "History Error:",
            e
        )

        return render_template(

            "history.html",

            predictions=[],

            error="Unable to load prediction history.",

            active_page="history"

        )


# =========================================================
# PREDICTION DETAILS
# =========================================================

@app.route(
    "/history/<int:prediction_id>"
)
def prediction_details(
    prediction_id
):

    try:

        conn = sqlite3.connect(
            DATABASE
        )

        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM predictions
            WHERE id = ?
        """, (
            prediction_id,
        ))

        prediction_record = cursor.fetchone()

        conn.close()

        # -------------------------------------------------
        # RECORD NOT FOUND
        # -------------------------------------------------

        if prediction_record is None:

            return render_template(

                "prediction_details.html",

                record=None,

                student_data={},

                skills=[],

                recommendations=[],

                shap_data=[],

                error="Assessment not found.",

                active_page="history"

            )

        # -------------------------------------------------
        # CONVERT DATABASE ROW
        # -------------------------------------------------

        record = dict(
            prediction_record
        )

        # -------------------------------------------------
        # LOAD STUDENT DATA
        # -------------------------------------------------

        try:

            student_data = (
                json.loads(
                    record["student_data"]
                )
                if record.get(
                    "student_data"
                )
                else {}
            )

        except Exception:

            student_data = {}

        # -------------------------------------------------
        # LOAD SKILLS
        # -------------------------------------------------

        try:

            skills = (
                json.loads(
                    record["skills"]
                )
                if record.get(
                    "skills"
                )
                else []
            )

        except Exception:

            skills = []

        # -------------------------------------------------
        # LOAD RECOMMENDATIONS
        # -------------------------------------------------

        try:

            recommendations = (
                json.loads(
                    record["recommendations"]
                )
                if record.get(
                    "recommendations"
                )
                else []
            )

        except Exception:

            recommendations = []

        # -------------------------------------------------
        # LOAD SHAP DATA
        # -------------------------------------------------

        try:

            shap_data = (
                json.loads(
                    record["shap_data"]
                )
                if record.get(
                    "shap_data"
                )
                else []
            )

        except Exception:

            shap_data = []

        # -------------------------------------------------
        # DISPLAY DETAILS
        # -------------------------------------------------

        return render_template(

            "prediction_details.html",

            record=record,

            student_data=student_data,

            skills=skills,

            recommendations=recommendations,

            shap_data=shap_data,

            active_page="history"

        )

    except Exception as e:

        print(
            "Prediction Details Error:",
            e
        )

        return render_template(

            "prediction_details.html",

            record=None,

            student_data={},

            skills=[],

            recommendations=[],

            shap_data=[],

            error="Unable to load assessment details.",

            active_page="history"

        )


# =========================================================
# ANALYTICS DASHBOARD
# =========================================================

@app.route("/analytics")
def analytics():

    try:

        conn = sqlite3.connect(
            DATABASE
        )

        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        # -------------------------------------------------
        # TOTAL ASSESSMENTS
        # -------------------------------------------------

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM predictions
        """)

        total_assessments = (
            cursor.fetchone()["total"]
        )

        # -------------------------------------------------
        # AVERAGE PROBABILITY
        # -------------------------------------------------

        cursor.execute("""
            SELECT AVG(probability) AS average
            FROM predictions
        """)

        average_result = (
            cursor.fetchone()["average"]
        )

        if average_result is None:

            average_probability = 0

        else:

            average_probability = round(
                float(average_result),
                2
            )

        # -------------------------------------------------
        # HIGH READINESS
        # -------------------------------------------------

        cursor.execute("""
            SELECT COUNT(*) AS count
            FROM predictions
            WHERE status = 'High Readiness'
        """)

        high_readiness = (
            cursor.fetchone()["count"]
        )

        # -------------------------------------------------
        # MODERATE READINESS
        # -------------------------------------------------

        cursor.execute("""
            SELECT COUNT(*) AS count
            FROM predictions
            WHERE status = 'Moderate Readiness'
        """)

        moderate_readiness = (
            cursor.fetchone()["count"]
        )

        # -------------------------------------------------
        # NEEDS IMPROVEMENT
        # -------------------------------------------------

        cursor.execute("""
            SELECT COUNT(*) AS count
            FROM predictions
            WHERE status = 'Needs Improvement'
        """)

        needs_improvement = (
            cursor.fetchone()["count"]
        )

        # -------------------------------------------------
        # PREDICTED PLACED
        # -------------------------------------------------

        cursor.execute("""
            SELECT COUNT(*) AS count
            FROM predictions
            WHERE prediction = '1'
               OR LOWER(prediction) = 'placed'
        """)

        predicted_placed = (
            cursor.fetchone()["count"]
        )

        # -------------------------------------------------
        # PREDICTED NOT PLACED
        # -------------------------------------------------

        cursor.execute("""
            SELECT COUNT(*) AS count
            FROM predictions
            WHERE prediction = '0'
               OR LOWER(prediction) = 'not placed'
        """)

        predicted_not_placed = (
            cursor.fetchone()["count"]
        )

        # -------------------------------------------------
        # RECENT ASSESSMENTS
        # -------------------------------------------------

        cursor.execute("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT 5
        """)

        recent_predictions = (
            cursor.fetchall()
        )

        conn.close()

        # -------------------------------------------------
        # PERCENTAGES
        # -------------------------------------------------

        if total_assessments > 0:

            placed_percentage = round(
                predicted_placed
                / total_assessments
                * 100,
                2
            )

            high_percentage = round(
                high_readiness
                / total_assessments
                * 100,
                2
            )

            moderate_percentage = round(
                moderate_readiness
                / total_assessments
                * 100,
                2
            )

            improvement_percentage = round(
                needs_improvement
                / total_assessments
                * 100,
                2
            )

        else:

            placed_percentage = 0
            high_percentage = 0
            moderate_percentage = 0
            improvement_percentage = 0

        # -------------------------------------------------
        # DISPLAY ANALYTICS
        # -------------------------------------------------

        return render_template(

            "analytics.html",

            total_assessments=
                total_assessments,

            average_probability=
                average_probability,

            predicted_placed=
                predicted_placed,

            predicted_not_placed=
                predicted_not_placed,

            placed_percentage=
                placed_percentage,

            high_readiness=
                high_readiness,

            moderate_readiness=
                moderate_readiness,

            needs_improvement=
                needs_improvement,

            high_percentage=
                high_percentage,

            moderate_percentage=
                moderate_percentage,

            improvement_percentage=
                improvement_percentage,

            recent_predictions=
                recent_predictions,

            active_page="analytics"

        )

    except Exception as e:

        print(
            "Analytics Error:",
            e
        )

        return render_template(

            "analytics.html",

            total_assessments=0,

            average_probability=0,

            predicted_placed=0,

            predicted_not_placed=0,

            placed_percentage=0,

            high_readiness=0,

            moderate_readiness=0,

            needs_improvement=0,

            high_percentage=0,

            moderate_percentage=0,

            improvement_percentage=0,

            recent_predictions=[],

            error="Unable to load analytics data.",

            active_page="analytics"

        )


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # =================================================
        # CHECK MODEL
        # =================================================

        if model is None:

            return render_template(

                "assessment.html",

                error=(
                    "The prediction model could not be loaded. "
                    "Please contact the administrator."
                ),

                active_page="assessment"

            )


        # =================================================
        # GET FORM VALUES
        # =================================================

        name = request.form.get(
            "name",
            ""
        ).strip()

        gender = request.form.get(
            "gender",
            ""
        ).strip()

        tenth_board = request.form.get(
            "tenth_board",
            ""
        ).strip()

        tenth_marks_raw = request.form.get(
            "tenth_marks",
            ""
        ).strip()

        twelfth_board = request.form.get(
            "twelfth_board",
            ""
        ).strip()

        twelfth_marks_raw = request.form.get(
            "twelfth_marks",
            ""
        ).strip()

        stream = request.form.get(
            "stream",
            ""
        ).strip()

        cgpa_raw = request.form.get(
            "cgpa",
            ""
        ).strip()

        internship = request.form.get(
            "internship",
            ""
        ).strip()

        training = request.form.get(
            "training",
            ""
        ).strip()

        backlog = request.form.get(
            "backlog",
            ""
        ).strip()

        project = request.form.get(
            "project",
            ""
        ).strip()

        communication_raw = request.form.get(
            "communication",
            ""
        ).strip()

        technical_course = request.form.get(
            "technical_course",
            ""
        ).strip()


        # =================================================
        # REQUIRED FIELD VALIDATION
        # =================================================

        required_values = {

            "Name":
                name,

            "Gender":
                gender,

            "10th Board":
                tenth_board,

            "10th Marks":
                tenth_marks_raw,

            "12th Board":
                twelfth_board,

            "12th Marks":
                twelfth_marks_raw,

            "Stream":
                stream,

            "CGPA":
                cgpa_raw,

            "Internship":
                internship,

            "Training":
                training,

            "Backlog":
                backlog,

            "Innovative Project":
                project,

            "Communication Level":
                communication_raw,

            "Technical Course":
                technical_course

        }


        missing_fields = [

            field_name

            for field_name, value
            in required_values.items()

            if value == ""

        ]


        if missing_fields:

            return render_template(

                "assessment.html",

                error=(
                    "Please fill in all required fields."
                ),

                active_page="assessment"

            )


        # =================================================
        # NUMERIC CONVERSION
        # =================================================

        try:

            tenth_marks = float(
                tenth_marks_raw
            )

            twelfth_marks = float(
                twelfth_marks_raw
            )

            cgpa = float(
                cgpa_raw
            )

            communication = int(
                communication_raw
            )

        except ValueError:

            return render_template(

                "assessment.html",

                error=(
                    "Please enter valid numeric values "
                    "for marks, CGPA and communication level."
                ),

                active_page="assessment"

            )


        # =================================================
        # RANGE VALIDATION
        # =================================================

        if not 0 <= tenth_marks <= 100:

            return render_template(

                "assessment.html",

                error=(
                    "10th marks must be between 0 and 100."
                ),

                active_page="assessment"

            )


        if not 0 <= twelfth_marks <= 100:

            return render_template(

                "assessment.html",

                error=(
                    "12th marks must be between 0 and 100."
                ),

                active_page="assessment"

            )


        if not 0 <= cgpa <= 10:

            return render_template(

                "assessment.html",

                error=(
                    "CGPA must be between 0 and 10."
                ),

                active_page="assessment"

            )


        if not 1 <= communication <= 4:

            return render_template(

                "assessment.html",

                error=(
                    "Communication level must be between 1 and 4."
                ),

                active_page="assessment"

            )


        # =================================================
        # STUDENT DATA
        # =================================================

        student_data = {

            "Gender":
                gender,

            "10th board":
                tenth_board,

            "10th marks":
                tenth_marks,

            "12th board":
                twelfth_board,

            "12th marks":
                twelfth_marks,

            "Stream":
                stream,

            "Cgpa":
                cgpa,

            "Internships(Y/N)":
                internship,

            "Training(Y/N)":
                training,

            "Backlog in 5th sem":
                backlog,

            "Innovative Project(Y/N)":
                project,

            "Communication level":
                communication,

            "Technical Course(Y/N)":
                technical_course

        }


        # =================================================
        # DATAFRAME
        # =================================================

        input_data = pd.DataFrame(
            [student_data]
        )


        # =================================================
        # ML PREDICTION
        # =================================================

        prediction = model.predict(
            input_data
        )[0]

        probability = (

            model.predict_proba(
                input_data
            )[0][1]

            * 100

        )

        probability = round(
            probability,
            2
        )


        # =================================================
        # READINESS STATUS
        # =================================================

        if probability >= 75:

            status = "High Readiness"

        elif probability >= 50:

            status = "Moderate Readiness"

        else:

            status = "Needs Improvement"


        # =================================================
        # SKILL GAPS
        # =================================================

        skills = []


        if student_data[
            "Cgpa"
        ] < 7.5:

            skills.append(
                "CGPA"
            )


        if student_data[
            "10th marks"
        ] < 70:

            skills.append(
                "10th Academic Performance"
            )


        if student_data[
            "12th marks"
        ] < 70:

            skills.append(
                "12th Academic Performance"
            )


        if student_data[
            "Communication level"
        ] < 3:

            skills.append(
                "Communication Skills"
            )


        if student_data[
            "Internships(Y/N)"
        ].lower() == "no":

            skills.append(
                "Internship Experience"
            )


        if student_data[
            "Training(Y/N)"
        ].lower() == "no":

            skills.append(
                "Training"
            )


        if student_data[
            "Innovative Project(Y/N)"
        ].lower() == "no":

            skills.append(
                "Projects"
            )


        if student_data[
            "Technical Course(Y/N)"
        ].lower() == "no":

            skills.append(
                "Technical Courses"
            )


        if student_data[
            "Backlog in 5th sem"
        ].lower() == "yes":

            skills.append(
                "Academic Backlogs"
            )


        # =================================================
        # RECOMMENDATIONS
        # =================================================

        recommendations = []


        if student_data[
            "Cgpa"
        ] < 7.5:

            recommendations.append(
                "Focus on improving your academic performance and CGPA."
            )


        if student_data[
            "Communication level"
        ] < 3:

            recommendations.append(
                "Practice communication, group discussions and HR interviews."
            )


        if student_data[
            "Internships(Y/N)"
        ].lower() == "no":

            recommendations.append(
                "Try to gain internship or industry experience."
            )


        if student_data[
            "Innovative Project(Y/N)"
        ].lower() == "no":

            recommendations.append(
                "Build an industry-oriented project and add it to your resume."
            )


        if student_data[
            "Technical Course(Y/N)"
        ].lower() == "no":

            recommendations.append(
                "Complete relevant technical courses or certifications."
            )


        if student_data[
            "Training(Y/N)"
        ].lower() == "no":

            recommendations.append(
                "Participate in technical training programs."
            )


        if student_data[
            "Backlog in 5th sem"
        ].lower() == "yes":

            recommendations.append(
                "Clear academic backlogs and maintain consistent performance."
            )


        if not recommendations:

            recommendations.append(
                "Excellent profile! Continue developing your technical and communication skills."
            )


        # =================================================
        # SHAP EXPLANATION
        # =================================================

        shap_data = []


        try:

            shap_df = get_shap_explanation(
                input_data
            )

            shap_df = shap_df.head(
                8
            )

            max_shap = (

                shap_df[
                    "shap_value"
                ]
                .abs()
                .max()

            )

            if max_shap == 0:

                max_shap = 1


            for _, row in shap_df.iterrows():

                value = float(
                    row[
                        "shap_value"
                    ]
                )

                percentage = (

                    abs(value)
                    / max_shap
                    * 100

                )

                direction = (

                    "positive"

                    if value >= 0

                    else "negative"

                )

                explanation_text = (
                    get_feature_explanation(
                        row["feature"],
                        value,
                        student_data
                    )
                )

                shap_data.append({

                    "feature":
                        row["feature"],

                    "value":
                        round(
                            value,
                            3
                        ),

                    "width":
                        round(
                            percentage,
                            1
                        ),

                    "direction":
                        direction,

                    "explanation":
                        explanation_text

                })


            print(
                "SHAP explanation generated successfully."
            )


        except Exception as shap_error:

            print(
                "SHAP Error:",
                shap_error
            )


        # =================================================
        # SAVE TO DATABASE
        # =================================================

        conn = sqlite3.connect(
            DATABASE
        )

        cursor = conn.cursor()


        cursor.execute("""

            INSERT INTO predictions

            (
                student_name,
                probability,
                status,
                prediction,
                assessment_date,
                student_data,
                skills,
                recommendations,
                shap_data
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

        """, (

            name,

            probability,

            status,

            str(
                prediction
            ),

            datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            ),

            json.dumps(
                student_data
            ),

            json.dumps(
                skills
            ),

            json.dumps(
                recommendations
            ),

            json.dumps(
                shap_data
            )

        ))


        conn.commit()
        conn.close()


        # =================================================
        # RESULT PAGE
        # =================================================

        return render_template(

            "result.html",

            name=name,

            probability=
                probability,

            status=
                status,

            prediction=
                prediction,

            skills=
                skills,

            recommendations=
                recommendations,

            data=
                student_data,

            shap_data=
                shap_data

        )


    except Exception as e:

        # =================================================
        # FRIENDLY PREDICTION ERROR
        # =================================================

        print(
            "Prediction Error:",
            e
        )

        return render_template(

            "assessment.html",

            error=(
                "Unable to process the assessment. "
                "Please check your inputs and try again."
            ),

            active_page="assessment"

        )


# =========================================================
# ABOUT
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html",
        active_page="about"
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)