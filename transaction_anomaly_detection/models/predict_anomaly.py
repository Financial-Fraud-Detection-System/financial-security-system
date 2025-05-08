import json
import logging
import os
import traceback

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, layers

# Define paths for model and artifacts
# The artifacts directory should be relative to the current file's location
MODEL_PATH = MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "artifacts", "isolation_forest_model.pkl"
)
MAPPING_PATH = os.path.join(
    os.path.dirname(__file__), "artifacts", "category_mappings.json"
)
AUTOENCODER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "artifacts", "autoencoder_model_scaled.keras"
)
SCALER_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "scaler.pkl")

# Set up logging
logger = logging.getLogger(__name__)


class Autoencoder(Model):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        hidden_dim_1,
        hidden_dim_2,
        name="autoencoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)  # Pass standard args like name

        # Store dimensions as attributes - needed for get_config AND building layers
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        # Build the encoder layers
        self.encoder = tf.keras.Sequential(
            [
                # Use InputLayer for explicit shape definition
                layers.InputLayer(input_shape=(self.input_dim,), name="enc_input"),
                layers.Dense(self.hidden_dim_2, activation="relu", name="enc_dense_1"),
                layers.BatchNormalization(name="enc_bn_1"),
                layers.Dropout(0.1, name="enc_dropout_1"),
                layers.Dense(self.hidden_dim_1, activation="relu", name="enc_dense_2"),
                layers.BatchNormalization(name="enc_bn_2"),
                layers.Dropout(0.1, name="enc_dropout_2"),
                layers.Dense(
                    self.encoding_dim, activation="relu", name="enc_dense_latent"
                ),
            ],
            name="encoder",
        )

        # Build the decoder layers
        self.decoder = tf.keras.Sequential(
            [
                # Use InputLayer for explicit shape definition
                layers.InputLayer(input_shape=(self.encoding_dim,), name="dec_input"),
                layers.Dense(self.hidden_dim_1, activation="relu", name="dec_dense_1"),
                layers.BatchNormalization(name="dec_bn_1"),
                layers.Dropout(0.1, name="dec_dropout_1"),
                layers.Dense(self.hidden_dim_2, activation="relu", name="dec_dense_2"),
                layers.BatchNormalization(name="dec_bn_2"),
                layers.Dropout(0.1, name="dec_dropout_2"),
                layers.Dense(self.input_dim, activation="linear", name="dec_output"),
            ],
            name="decoder",
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    # get_config should return arguments needed by __init__ plus any base config
    def get_config(self):
        # Get config from the parent class (Model)
        base_config = super().get_config()
        # Add the specific arguments needed to initialize *this* class
        config = {
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "hidden_dim_1": self.hidden_dim_1,
            "hidden_dim_2": self.hidden_dim_2,
        }
        # Combine base config with custom args
        base_config.update(config)
        return base_config

    # from_config reconstructs the object using the dictionary from get_config
    @classmethod
    def from_config(cls, config):
        # Log the received config for debugging
        logger.debug(f"Autoencoder.from_config received config: {config}")

        # First try to get values from the config
        init_args = {
            "input_dim": config.pop("input_dim", None),
            "encoding_dim": config.pop("encoding_dim", None),
            "hidden_dim_1": config.pop("hidden_dim_1", None),
            "hidden_dim_2": config.pop("hidden_dim_2", None),
            "name": config.pop("name", "autoencoder"),
        }

        # Check if any required args are missing
        required_args = ["input_dim", "encoding_dim", "hidden_dim_1", "hidden_dim_2"]
        missing_args = [arg for arg in required_args if init_args[arg] is None]

        # If any args are missing, use hardcoded values from training
        if missing_args:
            logger.warning(
                f"Missing args in config: {missing_args}. Using hardcoded values."
            )

            # Hardcoded values from your training
            default_values = {
                "input_dim": 22,
                "encoding_dim": 14,
                "hidden_dim_1": 21,
                "hidden_dim_2": 28,
            }

            # Fill in missing values with defaults
            for arg in missing_args:
                init_args[arg] = default_values[arg]

            logger.debug(f"Using dimensions for Autoencoder: {init_args}")

        return cls(**init_args, **config)


def load_model(model_path):
    """Load the trained Isolation Forest model."""
    logger.debug(f"Attempting to load model from path: '{model_path}'")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at specified path: '{model_path}'")
        raise FileNotFoundError(f"Model not found at {model_path}")

    try:
        logger.debug(f"Calling joblib.load('{model_path}')")
        loaded_model = joblib.load(model_path)
        logger.debug(f"Model loaded successfully. Object type: {type(loaded_model)}")
        if loaded_model is None:
            logger.error("CRITICAL: joblib.load returned None unexpectedly!")
            raise ValueError("Model loading resulted in None unexpectedly.")
        return loaded_model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
        raise


def load_category_mappings(mapping_path):
    """Load categorical encoding mappings from JSON."""
    logger.debug(f"Loading category mappings from {mapping_path}")
    if not os.path.exists(mapping_path):
        logger.error(f"Mapping file not found at specified path: '{mapping_path}'")
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}")

    try:
        with open(mapping_path, "r") as f:
            mappings = json.load(f)
        logger.debug("Category mappings loaded successfully.")
        return mappings
    except Exception as e:
        logger.error(f"Failed to load category mappings: {e}", exc_info=True)
        raise


def preprocess_input(data: dict, cat_mappings: dict):
    """Preprocess the input data dictionary and encode categorical columns."""
    logger.debug("Starting preprocessing...")
    try:
        # Create DataFrame from the single transaction dictionary
        df = pd.DataFrame([data])
        logger.debug(f"Initial DataFrame columns: {df.columns.tolist()}")

        # Apply category mappings
        for col, mapping in cat_mappings.items():
            if col in df.columns:
                original_value = df[col].iloc[0]  # Get value before mapping
                mapped_value = df[col].map(mapping).fillna(-1).astype(int)
                df[col] = mapped_value
                logger.debug(
                    f"Mapped column '{col}': '{original_value}' -> {mapped_value.iloc[0]}"
                )
            else:
                # If a mapped column is entirely missing from input, add it with default value
                logger.warning(
                    f"Column '{col}' expected by mappings not found in input data. Filling with -1."
                )
                df[col] = -1

        logger.info(f"Preprocessing complete. Final columns: {df.columns.tolist()}")
        logger.debug(f"Preprocessed data head:\n{df.head()}")
        return df

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise


def run_prediction(transaction_data: dict):
    """Runs prediction using the Isolation Forest model."""
    logger.debug("--- Starting Prediction Run (Isolation Forest) ---")
    try:
        # 1. Load model and mappings
        model = load_model(MODEL_PATH)
        cat_mappings = load_category_mappings(MAPPING_PATH)

        # 2. Preprocess the input dictionary
        processed_data = preprocess_input(transaction_data, cat_mappings)

        # 3. Align columns with model features (optional but recommended)
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is not None:
            current_features = processed_data.columns.tolist()
            missing_features = set(model_features) - set(current_features)
            if missing_features:
                logger.warning(
                    f"Missing features for Isolation Forest model: {missing_features}. Adding with default value 0."
                )
                for feature in missing_features:
                    processed_data[feature] = 0

            extra_features = set(current_features) - set(model_features)
            if extra_features:
                logger.warning(
                    f"Extra features in input data: {extra_features}. Removing them."
                )
                processed_data = processed_data.drop(columns=list(extra_features))

            processed_data = processed_data[model_features]
            logger.debug("Data columns aligned with Isolation Forest model features.")

        # Predict anomaly score
        logger.debug("Making prediction with the Isolation Forest model...")
        predict_input = (
            processed_data.values
            if isinstance(processed_data, pd.DataFrame)
            else np.array(processed_data).reshape(1, -1)
        )
        scores = model.decision_function(predict_input)
        threshold = -0.1
        prediction = 1 if scores[0] < threshold else 0
        is_anomaly = prediction == 1
        logger.info(
            f"Isolation Forest Prediction complete. Score: {scores[0]:.4f}, Threshold: {threshold}, Prediction: {prediction} (Anomaly={is_anomaly})"
        )

        return {
            "score": float(scores[0]),
            "prediction": int(prediction),
            "is_anomaly": bool(is_anomaly),
        }

    except FileNotFoundError as e:
        logger.error(f"Isolation Forest Prediction failed: File not found - {e}")
        return None
    except Exception as e:
        if "X does not have valid feature names" in str(e):
            logger.warning(f"Sklearn UserWarning during prediction: {e}")
        else:
            logger.error(
                f"An error occurred during the Isolation Forest prediction process: {e}",
                exc_info=True,
            )
        return None


def load_keras_model(model_path: str):
    """Load the trained Keras autoencoder model (using custom_objects)."""
    logger.debug(
        f"Attempting to load Keras model from path: '{model_path}' (using custom_objects)"
    )
    if not os.path.exists(model_path):
        logger.error(f"Keras model file not found at specified path: '{model_path}'")
        raise FileNotFoundError(f"Keras model not found at {model_path}")
    try:
        # Suppress TensorFlow INFO/WARNING messages, show only errors
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.get_logger().setLevel("ERROR")

        # Use custom_objects to inform Keras about the Autoencoder class.
        # This is necessary because the decorator method is not supported by the env's Keras version.
        custom_objects = {"Autoencoder": Autoencoder}
        loaded_model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects
        )

        logger.debug(
            f"Keras model loaded successfully from '{model_path}'. Type: {type(loaded_model)}"
        )
        if loaded_model is None:
            logger.error(
                "CRITICAL: tf.keras.models.load_model returned None unexpectedly!"
            )
            raise ValueError("Keras model loading resulted in None unexpectedly.")
        return loaded_model
    except Exception as e:
        # Log the full traceback for Keras loading errors
        logger.error(
            f"Failed to load Keras model from {model_path}: {e}\n{traceback.format_exc()}"
        )
        # Provide a more specific hint if it looks like a format incompatibility
        if "Unable to open file" in str(e) or "file format" in str(e).lower():
            logger.error(
                "Hint: This might indicate the .keras file format is incompatible with the installed TensorFlow/Keras version."
            )
        raise


def load_scaler(scaler_path: str):
    """Load the trained scaler object (e.g., StandardScaler)."""
    logger.debug(f"Attempting to load scaler from path: '{scaler_path}'")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found at specified path: '{scaler_path}'")
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    try:
        scaler = joblib.load(scaler_path)
        logger.debug(
            f"Scaler loaded successfully from '{scaler_path}'. Type: {type(scaler)}"
        )
        if scaler is None:
            logger.error("CRITICAL: joblib.load for scaler returned None unexpectedly!")
            raise ValueError("Scaler loading resulted in None unexpectedly.")
        return scaler
    except Exception as e:
        logger.error(f"Failed to load scaler from {scaler_path}: {e}", exc_info=True)
        raise


def preprocess_input_for_autoencoder(
    data: dict, cat_mappings: dict, scaler, model_feature_names: list
):
    """
    Preprocess the input data dictionary for the autoencoder.
    """
    logger.debug("Starting preprocessing for autoencoder...")
    try:
        input_df = create_initial_dataframe(data)
        temp_processed_df = apply_categorical_encoding(input_df, cat_mappings)
        df_for_scaling = construct_feature_dataframe(
            temp_processed_df, model_feature_names, cat_mappings
        )
        df_scaled = apply_scaling(df_for_scaling, scaler, model_feature_names)
        return df_scaled
    except Exception as e:
        logger.error(f"Preprocessing for autoencoder failed: {e}", exc_info=True)
        raise


def create_initial_dataframe(data: dict):
    input_df = pd.DataFrame([data])
    logger.debug(
        f"Initial input DataFrame for autoencoder columns: {input_df.columns.tolist()}"
    )
    return input_df


def apply_categorical_encoding(input_df: pd.DataFrame, cat_mappings: dict):
    temp_processed_df = pd.DataFrame(index=input_df.index)
    present_columns = set(input_df.columns)
    for mapped_col_name, mapping_dict in cat_mappings.items():
        if mapped_col_name in present_columns:
            temp_processed_df[mapped_col_name] = (
                input_df[mapped_col_name]
                .map(lambda x: mapping_dict.get(str(x), -1))
                .fillna(-1)
                .astype(int)
            )
    for col in present_columns:
        if col not in cat_mappings:
            try:
                temp_processed_df[col] = pd.to_numeric(input_df[col])
            except ValueError:
                logger.warning(
                    f"Column '{col}' not in mappings and couldn't be converted to numeric. Skipping."
                )
    return temp_processed_df


def construct_feature_dataframe(
    temp_processed_df: pd.DataFrame, model_feature_names: list, cat_mappings: dict
):
    df_for_scaling = pd.DataFrame(
        columns=model_feature_names, index=temp_processed_df.index
    )
    available_processed_cols = set(temp_processed_df.columns)
    for feature_name in model_feature_names:
        if feature_name in available_processed_cols:
            df_for_scaling[feature_name] = temp_processed_df[feature_name]
        else:
            if feature_name in cat_mappings:
                logger.warning(
                    f"Feature '{feature_name}' not in input data. Filling with default mapped value -1."
                )
                df_for_scaling[feature_name] = -1
            else:
                logger.warning(
                    f"Numerical feature '{feature_name}' not in input data. Filling with 0.0."
                )
                df_for_scaling[feature_name] = 0.0
    df_for_scaling = df_for_scaling.astype(float)
    return df_for_scaling


def apply_scaling(df_for_scaling: pd.DataFrame, scaler, model_feature_names: list):
    if not hasattr(scaler, "transform"):
        raise AttributeError("Scaler object does not have a 'transform' method.")
    scaled_values = scaler.transform(df_for_scaling.values)
    df_scaled = pd.DataFrame(
        scaled_values, columns=model_feature_names, index=df_for_scaling.index
    )
    return df_scaled


def run_autoencoder_prediction(transaction_data: dict):
    """
    Runs prediction using the Keras Autoencoder model.
    """
    logger.debug("--- Starting Prediction Run (Autoencoder) ---")
    try:
        autoencoder_model, cat_mappings, scaler, expected_model_features = (
            load_autoencoder_dependencies()
        )
        processed_data_df = preprocess_input_for_autoencoder(
            transaction_data,
            cat_mappings,
            scaler,
            model_feature_names=expected_model_features,
        )
        return make_autoencoder_prediction(autoencoder_model, processed_data_df)
    except Exception as e:
        logger.error(f"Autoencoder prediction failed: {e}", exc_info=True)
        return None


def load_autoencoder_dependencies():
    autoencoder_model = load_keras_model(AUTOENCODER_MODEL_PATH)
    cat_mappings = load_category_mappings(MAPPING_PATH)
    scaler = load_scaler(SCALER_PATH)
    expected_model_features = get_expected_model_features(scaler, autoencoder_model)
    return autoencoder_model, cat_mappings, scaler, expected_model_features


def get_expected_model_features(scaler, autoencoder_model):
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    elif hasattr(scaler, "n_features_in_"):
        raise ValueError(
            "Scaler lacks 'feature_names_in_'. Cannot reliably determine feature names."
        )
    else:
        raise ValueError(
            "Cannot determine expected features for the autoencoder model."
        )


def make_autoencoder_prediction(autoencoder_model, processed_data_df):
    input_for_keras_model = processed_data_df.values
    reconstructions = autoencoder_model.predict(input_for_keras_model, verbose=0)
    reconstruction_error = calculate_reconstruction_error(
        input_for_keras_model, reconstructions
    )
    return evaluate_autoencoder_prediction(reconstruction_error)


def calculate_reconstruction_error(input_data, reconstructions):
    mae = np.mean(np.abs(input_data - reconstructions), axis=1)
    return mae[0]


def evaluate_autoencoder_prediction(reconstruction_error):
    autoencoder_threshold = 99.5
    is_anomaly = reconstruction_error > autoencoder_threshold
    prediction = 1 if is_anomaly else 0
    logger.info(
        f"Autoencoder prediction complete. Reconstruction Error: {reconstruction_error:.6f}, Threshold: {autoencoder_threshold}, Prediction: {prediction} (Anomaly={is_anomaly})"
    )
    return {
        "reconstruction_error": float(reconstruction_error),
        "prediction": int(prediction),
        "is_anomaly": bool(is_anomaly),
    }


def anomaly_detection_wrapper(transaction: dict):
    """
    Wrapper function to run both models and return a combined result.

    This function will run the Isolation Forest and Autoencoder models
    on the provided transaction data and return a single result.
    """
    logger.debug("Starting fraud detection ensemble analysis...")

    # Run both models using the existing functions
    result_isoforest = run_prediction(transaction)
    result_autoencoder = run_autoencoder_prediction(transaction)

    # Create result object with the original transaction
    result = transaction.copy()

    # Get the individual predictions (default to 0 if model failed)
    isoforest_prediction = (
        result_isoforest.get("prediction", 0) if result_isoforest else 0
    )
    autoencoder_prediction = (
        result_autoencoder.get("prediction", 0) if result_autoencoder else 0
    )

    # Ensemble logic - consider it an anomaly if either model flags it
    is_anomaly = isoforest_prediction == 1 or autoencoder_prediction == 1

    # Add fraud result to the original transaction
    result["is_anomaly"] = is_anomaly

    return result


if __name__ == "__main__":
    # Sample transaction data for testing
    sample_transaction = {
        "cc_num": "2291163933867244",
        "merchant": "fraud_Kirlin and Sons",
        "category": "personal_care",
        "amt": 2.86,
        "first": "Jeff",
        "last": "Elliott",
        "gender": "M",
        "street": "351 Darlene Green",
        "city": "Anytown",
        "state": "CA",
        "zip": "12345",
        "lat": 37.7749,
        "long": -80.9355,
        "city_pop": 333497,
        "job": "Mechanical engineer",
        "dob": "1968-03-19",
        "unix_time": 1371816865,
        "merch_lat": 33.986391,
        "merch_long": -81.200714,
    }

    print(json.dumps(sample_transaction, indent=2))

    # Run anomaly detection wrapper
    anomaly_result = anomaly_detection_wrapper(sample_transaction)

    # Print the final result
    print("-" * 30)
    print("Anomaly Detection Result:")
    print(json.dumps(anomaly_result, indent=2))
