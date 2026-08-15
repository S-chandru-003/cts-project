from __future__ import annotations

import os
import sys
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import get_logger


logger = get_logger(__name__)


class DataTransformation:
    """
    Loads the final provider-level feature dataset and prepares
    it for machine learning.

    Input:
        data/processed/provider_features.csv

    Output:
        X_train
        X_test
        y_train
        y_test

    Responsibilities:
        - Load processed provider features
        - Validate the dataset
        - Remove duplicate columns
        - Separate target from features
        - Remove Provider identifier
        - Remove constant features
        - Perform stratified train/test split
        - Impute missing values
        - Scale numerical features
        - Encode categorical features
        - Save the fitted preprocessor
    """

    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:

        self.test_size = test_size
        self.random_state = random_state

        logger.info(
            "DataTransformation initialized."
        )

    # ============================================================
    # PROJECT PATH
    # ============================================================

    @staticmethod
    def _get_project_root() -> str:
        """
        Finds the project root based on the location of this file.

        Current file:
            src/components/data_transformation.py

        Project root:
            ../../
        """

        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
            )
        )

        return project_root

    # ============================================================
    # LOAD PROCESSED DATA
    # ============================================================

    def load_processed_data(self) -> pd.DataFrame:
        """
        Loads the final provider-level feature dataset.

        Expected file:

            data/processed/provider_features.csv
        """

        try:

            project_root = (
                self._get_project_root()
            )

            processed_file = os.path.join(
                project_root,
                "data",
                "processed",
                "provider_features.csv",
            )

            if not os.path.exists(
                processed_file
            ):

                raise FileNotFoundError(
                    "Processed feature file not found:\n"
                    f"{processed_file}\n\n"
                    "Run feature engineering first and "
                    "make sure provider_features.csv exists "
                    "inside data/processed."
                )

            logger.info(
                "Loading processed feature dataset: %s",
                processed_file,
            )

            dataframe = pd.read_csv(
                processed_file
            )

            logger.info(
                "Processed dataset loaded successfully."
            )

            logger.info(
                "Dataset shape: %s",
                dataframe.shape,
            )

            return dataframe

        except Exception as exc:

            logger.exception(
                "Failed to load processed feature dataset."
            )

            raise CustomException(
                exc,
                sys,
            ) from exc

    # ============================================================
    # VALIDATE INPUT
    # ============================================================

    @staticmethod
    def _validate_input(
        dataframe: pd.DataFrame,
    ) -> None:

        if dataframe is None:

            raise ValueError(
                "Input dataframe cannot be None."
            )

        if dataframe.empty:

            raise ValueError(
                "Input dataframe is empty."
            )

        required_columns = {
            "Provider",
            "PotentialFraud",
        }

        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:

            raise ValueError(
                "Required columns are missing: "
                f"{sorted(missing_columns)}"
            )

    # ============================================================
    # REMOVE DUPLICATE COLUMNS
    # ============================================================

    @staticmethod
    def _remove_duplicate_columns(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Removes duplicate column names.

        Example:
            If ReimbursementPerClaim accidentally appears twice,
            only the first occurrence is retained.
        """

        duplicate_mask = (
            dataframe.columns.duplicated(
                keep="first"
            )
        )

        duplicate_columns = (
            dataframe.columns[
                duplicate_mask
            ].tolist()
        )

        if duplicate_columns:

            logger.warning(
                "Duplicate columns removed: %s",
                duplicate_columns,
            )

        return dataframe.loc[
            :,
            ~duplicate_mask,
        ].copy()

    # ============================================================
    # PREPARE TARGET
    # ============================================================

    @staticmethod
    def _prepare_target(
        target: pd.Series,
    ) -> pd.Series:
        """
        Converts PotentialFraud into binary labels.

        No  -> 0
        Yes -> 1
        """

        target = (
            target
            .astype(str)
            .str.strip()
            .str.lower()
        )

        target_mapping = {
            "no": 0,
            "yes": 1,
            "0": 0,
            "1": 1,
            "non-fraud": 0,
            "nonfraud": 0,
            "fraud": 1,
        }

        mapped_target = target.map(
            target_mapping
        )

        unknown_values = (
            target[
                mapped_target.isna()
            ]
            .drop_duplicates()
            .tolist()
        )

        if unknown_values:

            raise ValueError(
                "Unknown PotentialFraud values found: "
                f"{unknown_values}"
            )

        return mapped_target.astype(int)

    # ============================================================
    # CLEAN FEATURES
    # ============================================================

    @staticmethod
    def _clean_features(
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Converts infinite numerical values into NaN.

        The preprocessing pipeline will handle NaN values later.
        """

        features = features.copy()

        features = features.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return features

    # ============================================================
    # REMOVE CONSTANT FEATURES
    # ============================================================

    @staticmethod
    def _remove_constant_features(
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Removes features containing only one unique value.

        Such features cannot help a model distinguish fraud
        from non-fraud.
        """

        constant_columns = [
            column
            for column in features.columns
            if features[column].nunique(
                dropna=False
            ) <= 1
        ]

        if constant_columns:

            logger.info(
                "Constant features removed: %s",
                constant_columns,
            )

            features = features.drop(
                columns=constant_columns
            )

        return features

    # ============================================================
    # BUILD PREPROCESSOR
    # ============================================================

    @staticmethod
    def _build_preprocessor(
        X_train: pd.DataFrame,
    ) -> ColumnTransformer:
        """
        Creates preprocessing pipelines.

        Numerical features:
            Median imputation
            StandardScaler

        Categorical features:
            Most-frequent imputation
            OneHotEncoder
        """

        numerical_columns = (
            X_train
            .select_dtypes(
                include=["number"]
            )
            .columns
            .tolist()
        )

        categorical_columns = (
            X_train
            .select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool",
                ]
            )
            .columns
            .tolist()
        )

        logger.info(
            "Numerical feature count: %d",
            len(numerical_columns),
        )

        logger.info(
            "Categorical feature count: %d",
            len(categorical_columns),
        )

        # --------------------------------------------------------
        # Numerical pipeline
        # --------------------------------------------------------

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler()
                ),
            ]
        )

        # --------------------------------------------------------
        # Categorical pipeline
        # --------------------------------------------------------

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers = []

        if numerical_columns:

            transformers.append(
                (
                    "numerical",
                    numerical_pipeline,
                    numerical_columns,
                )
            )

        if categorical_columns:

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns,
                )
            )

        if not transformers:

            raise ValueError(
                "No usable features found."
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

    # ============================================================
    # SAVE PREPROCESSOR
    # ============================================================

    def _save_preprocessor(
        self,
        preprocessor: ColumnTransformer,
    ) -> str:
        """
        Saves the fitted preprocessing pipeline.

        This exact preprocessor will later be reused
        when making predictions on new provider data.
        """

        project_root = (
            self._get_project_root()
        )

        artifact_directory = os.path.join(
            project_root,
            "artifacts",
        )

        os.makedirs(
            artifact_directory,
            exist_ok=True,
        )

        preprocessor_path = os.path.join(
            artifact_directory,
            "preprocessor.pkl",
        )

        joblib.dump(
            preprocessor,
            preprocessor_path,
        )

        logger.info(
            "Preprocessor saved to: %s",
            preprocessor_path,
        )

        return preprocessor_path

    # ============================================================
    # MAIN TRANSFORMATION METHOD
    # ============================================================

    def transform_from_processed_data(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        pd.Series,
        pd.Series,
    ]:
        """
        Complete transformation pipeline.

        The method reads provider_features.csv directly,
        so the model-training notebook does not depend on
        variables created inside EDA.ipynb.
        """

        try:

            logger.info(
                "Starting transformation from processed data."
            )

            # ----------------------------------------------------
            # Load final feature dataset
            # ----------------------------------------------------

            dataframe = (
                self.load_processed_data()
            )

            # ----------------------------------------------------
            # Validate
            # ----------------------------------------------------

            self._validate_input(
                dataframe
            )

            # ----------------------------------------------------
            # Remove duplicate columns
            # ----------------------------------------------------

            dataframe = (
                self._remove_duplicate_columns(
                    dataframe
                )
            )

            logger.info(
                "Shape after duplicate-column removal: %s",
                dataframe.shape,
            )

            # ----------------------------------------------------
            # Prepare target
            # ----------------------------------------------------

            y = self._prepare_target(
                dataframe[
                    "PotentialFraud"
                ]
            )

            # ----------------------------------------------------
            # Prepare features
            # ----------------------------------------------------

            X = dataframe.drop(
                columns=[
                    "PotentialFraud",
                ]
            )

            # ----------------------------------------------------
            # Provider is an identifier.
            #
            # We don't want the model to memorize provider IDs.
            # The behavioral provider features are retained.
            # ----------------------------------------------------

            X = X.drop(
                columns=[
                    "Provider",
                ],
                errors="ignore",
            )

            # ----------------------------------------------------
            # Replace infinity with NaN
            # ----------------------------------------------------

            X = self._clean_features(
                X
            )

            # ----------------------------------------------------
            # Remove constant features
            # ----------------------------------------------------

            X = (
                self._remove_constant_features(
                    X
                )
            )

            logger.info(
                "Final feature count before split: %d",
                X.shape[1],
            )

            # ----------------------------------------------------
            # Stratified train/test split
            # ----------------------------------------------------

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=y,
                )
            )

            logger.info(
                "Training samples: %d",
                len(X_train),
            )

            logger.info(
                "Testing samples: %d",
                len(X_test),
            )

            logger.info(
                "Training class distribution:\n%s",
                y_train.value_counts(),
            )

            logger.info(
                "Testing class distribution:\n%s",
                y_test.value_counts(),
            )

            # ----------------------------------------------------
            # Build preprocessing pipeline
            # ----------------------------------------------------

            preprocessor = (
                self._build_preprocessor(
                    X_train
                )
            )

            # ----------------------------------------------------
            # FIT ONLY ON TRAINING DATA
            #
            # This prevents test-data leakage.
            # ----------------------------------------------------

            X_train_transformed = (
                preprocessor.fit_transform(
                    X_train
                )
            )

            # ----------------------------------------------------
            # TRANSFORM TEST USING TRAIN-FITTED PREPROCESSOR
            # ----------------------------------------------------

            X_test_transformed = (
                preprocessor.transform(
                    X_test
                )
            )

            # ----------------------------------------------------
            # Convert to numpy
            # ----------------------------------------------------

            X_train_transformed = np.asarray(
                X_train_transformed,
                dtype=np.float64,
            )

            X_test_transformed = np.asarray(
                X_test_transformed,
                dtype=np.float64,
            )

            # ----------------------------------------------------
            # Save preprocessing pipeline
            # ----------------------------------------------------

            self._save_preprocessor(
                preprocessor
            )

            # ----------------------------------------------------
            # Final logging
            # ----------------------------------------------------

            logger.info(
                "Transformed training shape: %s",
                X_train_transformed.shape,
            )

            logger.info(
                "Transformed testing shape: %s",
                X_test_transformed.shape,
            )

            logger.info(
                "Transformation completed successfully."
            )

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
            )

        except Exception as exc:

            logger.exception(
                "Data transformation failed."
            )

            raise CustomException(
                exc,
                sys,
            ) from exc


# ================================================================
# MODULE TEST
# ================================================================

if __name__ == "__main__":

    transformer = DataTransformation()

    print(
        "DataTransformation module loaded successfully."
    )