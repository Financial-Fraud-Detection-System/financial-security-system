from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from models.predict_score import (
    predict_credit_score_bayesian,
    predict_credit_score_combined,
    predict_credit_score_xgb,
)


@pytest.fixture
def sample_input():
    return {
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


def test_predict_credit_score_xgb(sample_input):
    mock_model = MagicMock()
    mock_model.predict.return_value = [np.int64(1)]

    mock_encoder = MagicMock()
    mock_encoder.inverse_transform.return_value = ["Standard"]

    result = predict_credit_score_xgb(sample_input, mock_model, mock_encoder)

    assert "Credit_Score" in result
    assert result["Credit_Score"] == 1


def test_predict_credit_score_bayesian(sample_input):
    # Mock inference result
    mock_infer_result = MagicMock()
    mock_infer_result.values = [0.1, 0.7, 0.2]  # Most probable: index 1

    mock_bayesian_model = MagicMock()
    mock_bayesian_model.nodes.return_value = list(sample_input.keys())

    mock_age_disc = MagicMock()
    mock_income_disc = MagicMock()
    mock_age_disc.transform.return_value = [[1]]
    mock_income_disc.transform.return_value = [[1]]

    mock_encoder = MagicMock()
    mock_encoder.inverse_transform.return_value = ["Standard"]

    with patch(
        "models.predict_score.VariableElimination",
        return_value=MagicMock(query=MagicMock(return_value=mock_infer_result)),
    ):
        result = predict_credit_score_bayesian(
            sample_input,
            mock_bayesian_model,
            mock_age_disc,
            mock_income_disc,
            mock_encoder,
        )

    assert "Credit_Score_Bayesian" in result
    assert result["Credit_Score_Bayesian"] == 1


@patch("models.predict_score.load_xgb_classifier_model")
@patch("models.predict_score.load_credit_score_encoder")
@patch("models.predict_score.load_bayesian_model")
@patch("models.predict_score.load_discretizers")
@patch("models.predict_score.VariableElimination")
def test_predict_credit_score_combined(
    mock_variable_elimination,
    mock_load_discretizers,
    mock_load_bayesian_model,
    mock_load_encoder,
    mock_load_xgb,
    sample_input,
):
    # XGBoost mocks
    mock_xgb_model = MagicMock()
    mock_xgb_model.predict.return_value = [np.int64(2)]
    mock_load_xgb.return_value = mock_xgb_model

    mock_encoder = MagicMock()
    mock_encoder.inverse_transform.side_effect = lambda x: [
        "Good" if x[0] == 2 else "Standard" if x[0] == 1 else "Poor"
    ]
    mock_load_encoder.return_value = mock_encoder

    # Bayesian mocks
    mock_bayes_model = MagicMock()
    mock_bayes_model.nodes.return_value = list(sample_input.keys())
    mock_load_bayesian_model.return_value = mock_bayes_model

    mock_age_disc = MagicMock()
    mock_income_disc = MagicMock()
    mock_age_disc.transform.return_value = [[2]]
    mock_income_disc.transform.return_value = [[1]]
    mock_load_discretizers.return_value = (mock_age_disc, mock_income_disc)

    mock_infer_result = MagicMock()
    mock_infer_result.values = [0.1, 0.9, 0.0]  # Highest: index 1 ("Standard")
    mock_infer_engine = MagicMock()
    mock_infer_engine.query.return_value = mock_infer_result
    mock_variable_elimination.return_value = mock_infer_engine

    result = predict_credit_score_combined(sample_input)

    assert result["Credit_Score"] == "Standard"
