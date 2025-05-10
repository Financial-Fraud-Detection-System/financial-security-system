from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from .predict_score import (
    predict_credit_score_bayesian,
    predict_credit_score_combined,
    predict_credit_score_xgb,
)


@pytest.fixture
def mock_xgb_model():
    with patch("xgboost.XGBClassifier") as MockXGB, patch(
        "pickle.load"
    ) as mock_pickle_load:
        # Mock the XGBClassifier
        mock_model = MagicMock()
        mock_model.predict.return_value = [np.int32(1)]
        MockXGB.return_value = mock_model

        # Mock the label encoder
        mock_label_encoder = MagicMock()
        mock_label_encoder.inverse_transform.side_effect = lambda x: [
            "Standard" if i == 1 else "Good" for i in x
        ]

        # Set side effects for pickle.load
        mock_pickle_load.side_effect = [mock_label_encoder]

        yield mock_model


@pytest.fixture
def mock_bayesian_model():
    with patch("pickle.load") as mock_pickle_load, patch(
        "pgmpy.inference.VariableElimination.query"
    ) as mock_query:
        # Create mock model
        mock_model = MagicMock()
        mock_model.nodes.return_value = {"Age", "Annual_Income", "Credit_Score"}

        # Mock the query result
        mock_query_result = MagicMock()
        mock_query_result.values = np.array([0.1, 0.7, 0.2])
        mock_query.return_value = mock_query_result

        # Simulate model, age_discretizer, income_discretizer
        mock_age_disc = MagicMock()
        mock_age_disc.transform.return_value = np.array([[1]])

        mock_income_disc = MagicMock()
        mock_income_disc.transform.return_value = np.array([[2]])

        # Mock label encoder
        mock_label_encoder = MagicMock()
        mock_label_encoder.inverse_transform.side_effect = lambda x: [
            "Standard" if i == 1 else "Good" for i in x
        ]

        # Set side effects for pickle.load
        mock_pickle_load.side_effect = [
            mock_model,
            mock_age_disc,
            mock_income_disc,
            mock_label_encoder,
        ]

        yield mock_model


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


def test_predict_credit_score_xgb(mock_xgb_model):
    result = predict_credit_score_xgb(new_data)
    assert result["Credit_Score"] in [0, 1, 2]


def test_predict_credit_score_bayesian(mock_bayesian_model):
    result = predict_credit_score_bayesian(new_data)
    assert result["Credit_Score_Bayesian"] in [0, 1, 2]
