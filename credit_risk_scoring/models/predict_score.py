import logging
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from pgmpy.inference import VariableElimination

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_xgb_classifier_model():
    """
    Load the trained XGBoost classifier model.
    Returns:
        xgb.XGBClassifier: Loaded XGBoost classifier model
    """
    model_path = os.path.join(os.path.dirname(__file__), "artifacts", "xgb_model.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    logger.info("XGBoost model loaded successfully from %s", model_path)
    return model


def load_credit_score_encoder():
    """
    Load the label encoder for Credit_Score.

    Returns:
        LabelEncoder: Loaded label encoder
    """
    credit_score_encoder_path = os.path.join(
        os.path.dirname(__file__), "artifacts", "Credit_Score_label_encoder.pkl"
    )
    with open(credit_score_encoder_path, "rb") as f:
        credit_score_encoder = pickle.load(f)
    logger.info("Credit score label encoder loaded from %s", credit_score_encoder_path)
    return credit_score_encoder


def load_bayesian_model():
    """
    Load the trained Bayesian Network model.

    Returns:
        BayesianModel: Loaded Bayesian Network model
    """
    model_path = os.path.join(os.path.dirname(__file__), "artifacts", "bayes_model.pkl")
    with open(model_path, "rb") as f:
        bayesian_model = pickle.load(f)
    logger.info("Bayesian model loaded successfully from %s", model_path)
    return bayesian_model


def load_discretizers():
    """
    Load the discretizers for Age and Annual_Income.

    Returns:
        tuple: Tuple containing the loaded discretizers
    """
    age_path = os.path.join(
        os.path.dirname(__file__), "artifacts", "age_discretizer.pkl"
    )
    income_path = os.path.join(
        os.path.dirname(__file__), "artifacts", "income_discretizer.pkl"
    )

    with open(age_path, "rb") as f:
        age_disc = pickle.load(f)
    with open(income_path, "rb") as f:
        income_disc = pickle.load(f)

    logger.info(
        "Discretizers loaded: Age from %s, Income from %s", age_path, income_path
    )
    return age_disc, income_disc


def predict_credit_score_xgb(input_data, xgb_classifier_model, credit_score_encoder):
    """
    Load the trained XGBoost model and label encoders to predict the credit score for new data.

    Args:
        input_data (dict): Dictionary containing feature values for prediction
        xgb_classifier_model (xgb.XGBClassifier): Trained XGBoost model
        credit_score_encoder (LabelEncoder): Trained label encoder for Credit_Score

    Returns:
        dict: Input data with the predicted credit score added
    """
    logger.info("Running XGBoost prediction for input: %s", input_data)
    input_df = pd.DataFrame([input_data])

    prediction_encoded = xgb_classifier_model.predict(input_df)[0]

    if isinstance(prediction_encoded, np.integer):
        prediction_encoded = int(prediction_encoded)
    else:
        logger.error("XGBoost prediction not an integer: %s", prediction_encoded)
        raise ValueError("Prediction is not an integer type.")

    prediction_label = convert_predict_score_to_label(
        prediction_encoded, credit_score_encoder
    )

    credit_score = get_score_from_label(prediction_label)
    logger.info(
        "XGBoost predicted score: %s (label: %s) for input: %s",
        credit_score,
        prediction_label,
        input_data,
    )

    result = input_data.copy()
    result["Credit_Score"] = credit_score
    return result


def predict_credit_score_bayesian(
    input_data, bayesian_model, age_disc, income_disc, credit_score_encoder
):
    """
    Load the trained Bayesian Network model and discretizers to predict the credit score for new data.

    Args:
        input_data (dict): Dictionary containing feature values for prediction
        bayesian_model (BayesianModel): Trained Bayesian Network model
        age_disc (KBinsDiscretizer): Discretizer for Age
        income_disc (KBinsDiscretizer): Discretizer for Annual_Income
        credit_score_encoder (LabelEncoder): Trained label encoder for Credit_Score

    Returns:
        dict: Input data with the predicted credit score added
    """
    logger.info("Running Bayesian prediction for input: %s", input_data)
    infer = VariableElimination(bayesian_model)
    processed_input = input_data.copy()

    if "Age" in processed_input:
        age_val = pd.DataFrame([[processed_input["Age"]]], columns=["Age"])
        processed_input["Age"] = int(age_disc.transform(age_val)[0][0])

    if "Annual_Income" in processed_input:
        inc_val = pd.DataFrame(
            [[processed_input["Annual_Income"]]], columns=["Annual_Income"]
        )
        processed_input["Annual_Income"] = int(income_disc.transform(inc_val)[0][0])

    valid_nodes = set(bayesian_model.nodes())
    filtered_input = {k: v for k, v in processed_input.items() if k in valid_nodes}

    result = infer.query(variables=["Credit_Score"], evidence=filtered_input)
    probabilities = [result.values[i] for i in range(3)]
    prediction = int(np.argmax(probabilities))

    prediction_label = convert_predict_score_to_label(prediction, credit_score_encoder)
    credit_score = get_score_from_label(prediction_label)

    logger.info(
        "Bayesian predicted score: %s (label: %s, probs: %s) for input: %s",
        credit_score,
        prediction_label,
        probabilities,
        input_data,
    )

    result_dict = input_data.copy()
    result_dict["Credit_Score_Bayesian"] = credit_score
    return result_dict


def convert_predict_score_to_label(credit_score, credit_score_encoder):
    return credit_score_encoder.inverse_transform([credit_score])[0]


def get_score_from_label(label):
    if label == "Good":
        return 2
    elif label == "Standard":
        return 1
    elif label == "Poor":
        return 0
    else:
        logger.error("Invalid label in get_score_from_label: %s", label)
        raise ValueError("Invalid label. Expected 'Good', 'Standard', or 'Poor'.")


def predict_credit_score_combined(data):
    logger.info("Starting combined credit score prediction")
    xgb_model = load_xgb_classifier_model()
    encoder = load_credit_score_encoder()
    bayes_model = load_bayesian_model()
    age_disc, income_disc = load_discretizers()

    result1 = predict_credit_score_xgb(data, xgb_model, encoder)
    result2 = predict_credit_score_bayesian(
        data, bayes_model, age_disc, income_disc, encoder
    )

    score1 = result1["Credit_Score"]
    score2 = result2["Credit_Score_Bayesian"]
    mean_score = int((score1 + score2) / 2)

    if mean_score == 2:
        final_pred = "Good"
    elif mean_score == 1:
        final_pred = "Standard"
    elif mean_score == 0:
        final_pred = "Poor"
    else:
        logger.error("Invalid mean score computed: %s", mean_score)
        raise ValueError("Invalid mean score. Expected 0, 1, or 2.")

    logger.info(
        "Combined credit score: %s (from XGB: %s, Bayesian: %s)",
        final_pred,
        score1,
        score2,
    )

    combined_result = data.copy()
    combined_result["Credit_Score"] = final_pred
    return combined_result


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    new_data = {
        "Age": 50,
        "Occupation": 15,
        "Annual_Income": 100000.0,
        "Num_Bank_Accounts": 3,
        "Num_Credit_Card": 1,
        "Interest_Rate": 1,
        "Num_of_Loan": 9,
        "Num_of_Delayed_Payment": 0,
        "Auto Loan": 0,
        "Credit-Builder Loan": 0,
        "Debt Consolidation Loan": 0,
        "Home Equity Loan": 0,
        "Mortgage Loan": 0,
        "Not Specified": 0,
        "Payday Loan": 0,
        "Personal Loan": 0,
        "Student Loan": 0,
    }

    try:
        combined_result = predict_credit_score_combined(new_data)
        print(f"Combined Predicted Credit Score: {combined_result['Credit_Score']}")
        print("Combined Result:")
        print(combined_result)
    except Exception as e:
        print("Prediction failed: %s", str(e))
