import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from pgmpy.inference import VariableElimination


def load_xgb_classifier_model():
    """
    Load the trained XGBoost classifier model.
    Returns:
        xgb.XGBClassifier: Loaded XGBoost classifier model
    """
    # Define paths
    model_path = os.path.join(os.path.dirname(__file__), "artifacts", "xgb_model.json")

    # Load the trained model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    return model


def load_credit_score_encoder():
    """
    Load the label encoder for Credit_Score.

    Returns:
        LabelEncoder: Loaded label encoder
    """
    # Define path
    credit_score_encoder_path = os.path.join(
        os.path.dirname(__file__), "artifacts", "Credit_Score_label_encoder.pkl"
    )

    # Load the Credit_Score label encoder
    with open(credit_score_encoder_path, "rb") as f:
        credit_score_encoder = pickle.load(f)

    return credit_score_encoder


def load_bayesian_model():
    """
    Load the trained Bayesian Network model.

    Returns:
        BayesianModel: Loaded Bayesian Network model
    """
    # Define path
    model_path = os.path.join(os.path.dirname(__file__), "artifacts", "bayes_model.pkl")

    # Load the trained model
    with open(model_path, "rb") as f:
        bayesian_model = pickle.load(f)

    return bayesian_model


def load_discretizers():
    """
    Load the discretizers for Age and Annual_Income.

    Returns:
        tuple: Tuple containing the loaded discretizers
    """
    # Define paths
    age_discretizer_path = os.path.join(
        os.path.dirname(__file__), "artifacts", "age_discretizer.pkl"
    )
    income_discretizer_path = os.path.join(
        os.path.dirname(__file__), "artifacts", "income_discretizer.pkl"
    )

    # Load the discretizers
    with open(age_discretizer_path, "rb") as f:
        age_disc = pickle.load(f)

    with open(income_discretizer_path, "rb") as f:
        income_disc = pickle.load(f)

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
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction_encoded = xgb_classifier_model.predict(input_df)[0]

    # Convert NumPy type to standard Python type
    if isinstance(prediction_encoded, np.integer):
        prediction_encoded = int(prediction_encoded)
    else:
        raise ValueError("Prediction is not an integer type.")

    # Convert prediction to original label
    prediction_label = convert_predict_score_to_label(
        prediction_encoded, credit_score_encoder
    )

    credit_score = get_score_from_label(prediction_label)

    # Add prediction to the input data
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
    # Create inference engine
    infer = VariableElimination(bayesian_model)

    # Process input data for Bayesian model
    processed_input = input_data.copy()

    # Discretize Age
    if "Age" in processed_input:
        age_val = pd.DataFrame([[processed_input["Age"]]], columns=["Age"])
        processed_input["Age"] = int(age_disc.transform(age_val)[0][0])

    # Discretize Annual_Income
    if "Annual_Income" in processed_input:
        inc_val = pd.DataFrame(
            [[processed_input["Annual_Income"]]], columns=["Annual_Income"]
        )
        processed_input["Annual_Income"] = int(income_disc.transform(inc_val)[0][0])

    # Filter evidence to match the model's nodes
    valid_nodes = set(bayesian_model.nodes())
    filtered_input = {k: v for k, v in processed_input.items() if k in valid_nodes}

    # Perform inference
    result = infer.query(variables=["Credit_Score"], evidence=filtered_input)

    # Get the highest probability class (0, 1, or 2)
    probabilities = [result.values[i] for i in range(3)]
    prediction = int(np.argmax(probabilities))

    # Convert prediction to original label
    prediction_label = convert_predict_score_to_label(prediction, credit_score_encoder)

    credit_score = get_score_from_label(prediction_label)

    # Add prediction to the input data
    result_dict = input_data.copy()
    result_dict["Credit_Score_Bayesian"] = credit_score

    return result_dict


def convert_predict_score_to_label(credit_score, credit_score_encoder):
    # # Convert prediction back to original label
    prediction_decoded = credit_score_encoder.inverse_transform([credit_score])[0]

    return prediction_decoded


def get_score_from_label(label):
    if label == "Good":
        return 2
    elif label == "Standard":
        return 1
    elif label == "Poor":
        return 0
    else:
        raise ValueError("Invalid label. Expected 'Good', 'Standard', or 'Poor'.")


def predict_credit_score_combined(data):
    # Load models and encoders
    xgb_classifier_model = load_xgb_classifier_model()
    credit_score_encoder = load_credit_score_encoder()
    bayesian_model = load_bayesian_model()
    age_disc, income_disc = load_discretizers()

    # Call the first prediction model
    result1 = predict_credit_score_xgb(data, xgb_classifier_model, credit_score_encoder)

    # Call the Bayesian prediction model
    result2 = predict_credit_score_bayesian(
        data, bayesian_model, age_disc, income_disc, credit_score_encoder
    )

    # Extract credit scores from both results
    score1 = result1["Credit_Score"]
    score2 = result2["Credit_Score_Bayesian"]

    # Calculate mean and floor down to integer
    mean_score = int((score1 + score2) / 2)

    if mean_score == 2:
        final_pred = "Good"
    elif mean_score == 1:
        final_pred = "Standard"
    elif mean_score == 0:
        final_pred = "Poor"
    else:
        raise ValueError("Invalid mean score. Expected 0, 1, or 2.")

    # Create combined result (copy of input data with added score)
    combined_result = data.copy()
    combined_result["Credit_Score"] = final_pred

    return combined_result


# Example usage
if __name__ == "__main__":
    # Sample input data
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

    # # Get prediction
    # result = predict_credit_score_xgb(new_data)
    # print(f"Predicted Credit Score: {result['Credit_Score']}")
    # print("Complete Result:")
    # print(result)

    # bayes_result = predict_credit_score_bayesian(new_data)
    # print(f"Bayesian Predicted Credit Score: {bayes_result['Credit_Score_Bayesian']}")

    combined_result = predict_credit_score_combined(new_data)
    print(f"Combined Predicted Credit Score: {combined_result['Credit_Score']}")
    print("Combined Result:")
    print(combined_result)
