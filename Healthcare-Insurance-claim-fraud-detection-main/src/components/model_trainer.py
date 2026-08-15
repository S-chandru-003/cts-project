from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import get_logger


warnings.filterwarnings("ignore")

logger = get_logger(__name__)


class ModelTraining:
    """
    Provider-level healthcare fraud model training.

    Design:
        - 5-fold stratified cross-validation
        - fold-safe missing-value imputation
        - SMOTE only inside the CV pipeline for selected models
        - probability-based model tuning with PR-AUC
        - out-of-fold threshold optimization using F1
        - untouched test-set evaluation
        - model selection WITHOUT using test metrics
        - artifact saving for deployment

    CatBoost is included as an additional tabular boosting candidate.

    Important:
        The final test set is used only for evaluation.
        It is never used to select hyperparameters or the threshold.
    """

    def __init__(
        self,
        random_state: int = 42,
        cv_folds: int = 5,
        n_iter: int = 60,
    ) -> None:

        self.random_state = random_state
        self.cv_folds = cv_folds
        self.n_iter = n_iter

        self.cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )

        logger.info("ModelTraining initialized.")

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_input(
        X_train: Any,
        X_test: Any,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:

        if X_train is None or X_test is None:
            raise ValueError(
                "Training or testing features cannot be None."
            )

        if y_train is None or y_test is None:
            raise ValueError(
                "Training or testing targets cannot be None."
            )

        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train have different lengths."
            )

        if len(X_test) != len(y_test):
            raise ValueError(
                "X_test and y_test have different lengths."
            )

        if len(np.unique(y_train)) < 2:
            raise ValueError(
                "Training data must contain both classes."
            )

        if len(np.unique(y_test)) < 2:
            raise ValueError(
                "Testing data must contain both classes."
            )

    # ============================================================
    # IMBALANCE
    # ============================================================

    @staticmethod
    def _calculate_scale_pos_weight(
        y: pd.Series,
    ) -> float:

        negative_count = int((y == 0).sum())
        positive_count = int((y == 1).sum())

        if positive_count == 0:
            raise ValueError(
                "No positive fraud samples found."
            )

        return negative_count / positive_count

    # ============================================================
    # THRESHOLD OPTIMIZATION
    # ============================================================

    @staticmethod
    def _find_best_threshold(
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> Tuple[float, float, float, float]:

        """
        Optimize F1 using ONLY out-of-fold training predictions.

        The final test set is never used here.
        """

        thresholds = np.arange(
            0.10,
            0.91,
            0.005,
        )

        best_threshold = 0.50
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0

        for threshold in thresholds:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            precision = precision_score(
                y_true,
                predictions,
                zero_division=0,
            )

            recall = recall_score(
                y_true,
                predictions,
                zero_division=0,
            )

            f1 = f1_score(
                y_true,
                predictions,
                zero_division=0,
            )

            if (
                f1 > best_f1
                or (
                    np.isclose(f1, best_f1)
                    and precision > best_precision
                )
            ):

                best_threshold = threshold
                best_precision = precision
                best_recall = recall
                best_f1 = f1

        return (
            float(best_threshold),
            float(best_precision),
            float(best_recall),
            float(best_f1),
        )

    # ============================================================
    # METRICS
    # ============================================================

    @staticmethod
    def _calculate_metrics(
        y_true: np.ndarray,
        probabilities: np.ndarray,
        threshold: float,
    ) -> Dict[str, float]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        return {
            "Precision": precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            "F1": f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            "PR_AUC": average_precision_score(
                y_true,
                probabilities,
            ),
            "ROC_AUC": roc_auc_score(
                y_true,
                probabilities,
            ),
        }

    # ============================================================
    # MODEL DEFINITIONS
    # ============================================================

    def _build_models(
        self,
        y_train: pd.Series,
    ) -> Dict[str, Tuple[Any, Dict[str, list]]]:

        scale_pos_weight = (
            self._calculate_scale_pos_weight(
                y_train
            )
        )

        logger.info(
            "Calculated imbalance ratio: %.3f",
            scale_pos_weight,
        )

        models = {}

        # ========================================================
        # LOGISTIC REGRESSION
        #
        # Imputer -> SMOTE -> Scaler -> LogisticRegression
        #
        # Everything is inside the CV pipeline.
        # ========================================================

        logistic_pipeline = ImbPipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=self.random_state,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        logistic_params = {
            "smote__sampling_strategy": [
                0.50,
                0.75,
                1.00,
            ],
            "model__C": [
                0.01,
                0.05,
                0.1,
                0.5,
                1.0,
                2.0,
                5.0,
            ],
            "model__class_weight": [
                None,
                "balanced",
            ],
            "model__solver": [
                "lbfgs",
                "liblinear",
            ],
        }

        models["Logistic Regression"] = (
            logistic_pipeline,
            logistic_params,
        )

        # ========================================================
        # RANDOM FOREST
        #
        # Imputer -> SMOTE -> RandomForest
        # ========================================================

        random_forest_pipeline = ImbPipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=self.random_state,
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        random_forest_params = {
            "smote__sampling_strategy": [
                0.50,
                0.75,
                1.00,
            ],
            "model__n_estimators": [
                300,
                500,
                700,
                900,
            ],
            "model__max_depth": [
                6,
                8,
                12,
                16,
                20,
                None,
            ],
            "model__min_samples_split": [
                2,
                5,
                10,
                20,
                30,
            ],
            "model__min_samples_leaf": [
                1,
                2,
                5,
                10,
                15,
            ],
            "model__max_features": [
                "sqrt",
                "log2",
                0.5,
                0.7,
            ],
            "model__class_weight": [
                None,
                "balanced",
                "balanced_subsample",
            ],
        }

        models["Random Forest"] = (
            random_forest_pipeline,
            random_forest_params,
        )

        # ========================================================
        # XGBOOST
        #
        # Imputer -> XGBoost
        # ========================================================

        xgb_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        random_state=self.random_state,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        )

        xgb_params = {
            "model__n_estimators": [
                200,
                400,
                600,
                800,
                1000,
            ],
            "model__max_depth": [
                2,
                3,
                4,
                5,
                6,
                8,
            ],
            "model__learning_rate": [
                0.01,
                0.02,
                0.03,
                0.05,
                0.08,
                0.10,
            ],
            "model__subsample": [
                0.65,
                0.75,
                0.85,
                0.95,
                1.00,
            ],
            "model__colsample_bytree": [
                0.60,
                0.70,
                0.80,
                0.90,
                1.00,
            ],
            "model__min_child_weight": [
                1,
                3,
                5,
                10,
                15,
            ],
            "model__gamma": [
                0,
                0.1,
                0.3,
                0.5,
                1.0,
            ],
            "model__reg_alpha": [
                0,
                0.01,
                0.1,
                0.5,
                1.0,
            ],
            "model__reg_lambda": [
                1,
                2,
                5,
                10,
                20,
            ],
            "model__scale_pos_weight": [
                scale_pos_weight * 0.50,
                scale_pos_weight * 0.75,
                scale_pos_weight,
                scale_pos_weight * 1.25,
                scale_pos_weight * 1.50,
            ],
        }

        models["XGBoost"] = (
            xgb_pipeline,
            xgb_params,
        )

        # ========================================================
        # CATBOOST
        #
        # Imputer -> CatBoost
        #
        # CatBoost is included because it is a strong gradient
        # boosting model for structured/tabular data.
        # ========================================================

        catboost_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    CatBoostClassifier(
                        loss_function="Logloss",
                        eval_metric="AUC",
                        random_seed=self.random_state,
                        verbose=False,
                        thread_count=-1,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )

        catboost_params = {
            "model__iterations": [
                300,
                500,
                700,
                900,
                1200,
            ],
            "model__depth": [
                4,
                5,
                6,
                7,
                8,
                10,
            ],
            "model__learning_rate": [
                0.01,
                0.02,
                0.03,
                0.05,
                0.08,
                0.10,
            ],
            "model__l2_leaf_reg": [
                1,
                3,
                5,
                7,
                10,
                15,
                20,
            ],
            "model__random_strength": [
                0,
                0.25,
                0.5,
                1,
                2,
            ],
            "model__bagging_temperature": [
                0,
                0.25,
                0.5,
                1,
                2,
            ],
            "model__border_count": [
                32,
                64,
                128,
                254,
            ],
            "model__auto_class_weights": [
                None,
                "Balanced",
            ],
        }

        models["CatBoost"] = (
            catboost_pipeline,
            catboost_params,
        )

        # ========================================================
        # LIGHTGBM
        #
        # Imputer -> LightGBM
        #
        # This is the main model we expect to perform best.
        # ========================================================

        lgbm_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        random_state=self.random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )

        lgbm_params = {
            "model__n_estimators": [
                200,
                400,
                600,
                800,
                1000,
                1200,
            ],
            "model__num_leaves": [
                7,
                15,
                23,
                31,
                40,
                50,
                70,
            ],
            "model__max_depth": [
                3,
                5,
                7,
                9,
                12,
                -1,
            ],
            "model__learning_rate": [
                0.005,
                0.01,
                0.02,
                0.03,
                0.05,
                0.08,
                0.10,
            ],
            "model__min_child_samples": [
                10,
                20,
                30,
                40,
                50,
                75,
                100,
            ],
            "model__subsample": [
                0.65,
                0.75,
                0.85,
                0.95,
                1.00,
            ],
            "model__colsample_bytree": [
                0.60,
                0.70,
                0.80,
                0.90,
                1.00,
            ],
            "model__reg_alpha": [
                0,
                0.01,
                0.05,
                0.1,
                0.5,
                1.0,
            ],
            "model__reg_lambda": [
                0,
                0.5,
                1,
                2,
                5,
                10,
                20,
            ],
            "model__scale_pos_weight": [
                scale_pos_weight * 0.50,
                scale_pos_weight * 0.75,
                scale_pos_weight,
                scale_pos_weight * 1.25,
                scale_pos_weight * 1.50,
                scale_pos_weight * 2.00,
            ],
        }

        models["LightGBM"] = (
            lgbm_pipeline,
            lgbm_params,
        )

        return models

    # ============================================================
    # HYPERPARAMETER TUNING
    # ============================================================

    def _tune_model(
        self,
        model_name: str,
        estimator: Any,
        param_grid: Dict[str, list],
        X_train: np.ndarray,
        y_train: pd.Series,
    ) -> Any:

        logger.info(
            "Starting hyperparameter tuning for %s.",
            model_name,
        )

        # PR-AUC is used for hyperparameter search.
        # This is more appropriate than optimizing a fixed
        # threshold on an imbalanced fraud dataset.
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=self.n_iter,
            scoring="average_precision",
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        search.fit(
            X_train,
            y_train,
        )

        logger.info(
            "%s best CV PR-AUC: %.6f",
            model_name,
            search.best_score_,
        )

        logger.info(
            "%s best parameters: %s",
            model_name,
            search.best_params_,
        )

        return search

    # ============================================================
    # OUT-OF-FOLD PREDICTIONS
    # ============================================================

    def _get_oof_predictions(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: pd.Series,
    ) -> np.ndarray:

        """
        Generate OOF probabilities from the training set.

        These predictions are used for threshold selection.
        The final test set is never touched.
        """

        logger.info(
            "Generating out-of-fold predictions."
        )

        oof_probabilities = cross_val_predict(
            estimator,
            X_train,
            y_train,
            cv=self.cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        return oof_probabilities

    # ============================================================
    # TRAIN / TEST EVALUATION
    # ============================================================

    def _evaluate_model(
        self,
        model_name: str,
        estimator: Any,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series,
        threshold: float,
        cv_pr_auc: float,
        cv_threshold_f1: float,
        cv_threshold_precision: float,
        cv_threshold_recall: float,
    ) -> Dict[str, Any]:

        train_probabilities = (
            estimator.predict_proba(X_train)[:, 1]
        )

        test_probabilities = (
            estimator.predict_proba(X_test)[:, 1]
        )

        train_metrics = self._calculate_metrics(
            y_train,
            train_probabilities,
            threshold,
        )

        test_metrics = self._calculate_metrics(
            y_test,
            test_probabilities,
            threshold,
        )

        f1_gap = (
            train_metrics["F1"]
            - test_metrics["F1"]
        )

        return {
            "Model": model_name,

            "CV_PR_AUC": cv_pr_auc,
            "CV_Threshold_F1": cv_threshold_f1,
            "CV_Threshold_Precision":
                cv_threshold_precision,
            "CV_Threshold_Recall":
                cv_threshold_recall,

            "Train_Precision":
                train_metrics["Precision"],
            "Train_Recall":
                train_metrics["Recall"],
            "Train_F1":
                train_metrics["F1"],
            "Train_PR_AUC":
                train_metrics["PR_AUC"],
            "Train_ROC_AUC":
                train_metrics["ROC_AUC"],

            "Test_Precision":
                test_metrics["Precision"],
            "Test_Recall":
                test_metrics["Recall"],
            "Test_F1":
                test_metrics["F1"],
            "Test_PR_AUC":
                test_metrics["PR_AUC"],
            "Test_ROC_AUC":
                test_metrics["ROC_AUC"],

            "BestThreshold": threshold,

            "F1_Gap": f1_gap,
        }

    # ============================================================
    # SAVE ARTIFACTS
    # ============================================================

    @staticmethod
    def _get_artifact_directory() -> str:

        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
            )
        )

        artifact_directory = os.path.join(
            project_root,
            "artifacts",
        )

        os.makedirs(
            artifact_directory,
            exist_ok=True,
        )

        return artifact_directory

    @classmethod
    def _save_artifacts(
        cls,
        best_model: Any,
        best_model_name: str,
        best_threshold: float,
        results_df: pd.DataFrame,
    ) -> None:

        artifact_directory = (
            cls._get_artifact_directory()
        )

        model_path = os.path.join(
            artifact_directory,
            "best_model.pkl",
        )

        threshold_path = os.path.join(
            artifact_directory,
            "fraud_threshold.pkl",
        )

        results_path = os.path.join(
            artifact_directory,
            "model_results.csv",
        )

        metadata_path = os.path.join(
            artifact_directory,
            "model_metadata.pkl",
        )

        joblib.dump(
            best_model,
            model_path,
        )

        joblib.dump(
            best_threshold,
            threshold_path,
        )

        results_df.to_csv(
            results_path,
            index=False,
        )

        metadata = {
            "best_model_name": best_model_name,
            "fraud_threshold": best_threshold,
        }

        joblib.dump(
            metadata,
            metadata_path,
        )

        logger.info(
            "Best model saved to: %s",
            model_path,
        )

        logger.info(
            "Threshold saved to: %s",
            threshold_path,
        )

        logger.info(
            "Results saved to: %s",
            results_path,
        )

    # ============================================================
    # MAIN TRAINING METHOD
    # ============================================================

    def train_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[Any, pd.DataFrame, float]:

        try:

            logger.info(
                "Starting complete model training pipeline."
            )

            # ----------------------------------------------------
            # Validate
            # ----------------------------------------------------

            self._validate_input(
                X_train,
                X_test,
                y_train,
                y_test,
            )

            # ----------------------------------------------------
            # Convert inputs
            # ----------------------------------------------------

            X_train = np.asarray(
                X_train,
                dtype=np.float64,
            )

            X_test = np.asarray(
                X_test,
                dtype=np.float64,
            )

            y_train = pd.Series(
                y_train
            ).astype(int)

            y_test = pd.Series(
                y_test
            ).astype(int)

            logger.info(
                "X_train shape: %s",
                X_train.shape,
            )

            logger.info(
                "X_test shape: %s",
                X_test.shape,
            )

            # ----------------------------------------------------
            # Build models
            # ----------------------------------------------------

            models = self._build_models(
                y_train
            )

            results = []
            trained_models = {}

            # ----------------------------------------------------
            # Train candidate models
            # ----------------------------------------------------

            for (
                model_name,
                (
                    estimator,
                    parameter_grid,
                ),
            ) in models.items():

                logger.info("=" * 70)
                logger.info(
                    "Training model: %s",
                    model_name,
                )

                # ------------------------------------------------
                # Hyperparameter tuning
                # ------------------------------------------------

                search = self._tune_model(
                    model_name,
                    estimator,
                    parameter_grid,
                    X_train,
                    y_train,
                )

                best_estimator = (
                    search.best_estimator_
                )

                # ------------------------------------------------
                # OOF probabilities
                # ------------------------------------------------

                oof_probabilities = (
                    self._get_oof_predictions(
                        best_estimator,
                        X_train,
                        y_train,
                    )
                )

                oof_pr_auc = average_precision_score(
                    y_train,
                    oof_probabilities,
                )

                # ------------------------------------------------
                # Threshold optimization
                # ------------------------------------------------

                (
                    best_threshold,
                    threshold_precision,
                    threshold_recall,
                    threshold_f1,
                ) = self._find_best_threshold(
                    y_train.to_numpy(),
                    oof_probabilities,
                )

                logger.info(
                    "%s OOF PR-AUC: %.4f",
                    model_name,
                    oof_pr_auc,
                )

                logger.info(
                    "%s threshold: %.3f",
                    model_name,
                    best_threshold,
                )

                logger.info(
                    "%s OOF threshold precision: %.4f",
                    model_name,
                    threshold_precision,
                )

                logger.info(
                    "%s OOF threshold recall: %.4f",
                    model_name,
                    threshold_recall,
                )

                logger.info(
                    "%s OOF threshold F1: %.4f",
                    model_name,
                    threshold_f1,
                )

                # ------------------------------------------------
                # Final fit on ALL training data
                # ------------------------------------------------

                best_estimator.fit(
                    X_train,
                    y_train,
                )

                # ------------------------------------------------
                # Evaluate
                # ------------------------------------------------

                model_results = (
                    self._evaluate_model(
                        model_name=model_name,
                        estimator=best_estimator,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        threshold=best_threshold,
                        cv_pr_auc=oof_pr_auc,
                        cv_threshold_f1=threshold_f1,
                        cv_threshold_precision=threshold_precision,
                        cv_threshold_recall=threshold_recall,
                    )
                )

                results.append(
                    model_results
                )

                trained_models[
                    model_name
                ] = {
                    "model": best_estimator,
                    "threshold": best_threshold,
                    "cv_pr_auc": oof_pr_auc,
                    "cv_threshold_f1": threshold_f1,
                }

            # ----------------------------------------------------
            # Results dataframe
            # ----------------------------------------------------

            results_df = pd.DataFrame(
                results
            )

            # ----------------------------------------------------
            # IMPORTANT:
            # Select using TRAINING CV information only.
            #
            # The final test metrics are NOT used for model
            # selection.
            #
            # Primary:
            #     CV threshold F1
            #
            # Secondary:
            #     CV PR-AUC
            #
            # Tertiary:
            #     CV threshold precision
            # ----------------------------------------------------

            results_df = (
                results_df
                .sort_values(
                    by=[
                        "CV_Threshold_F1",
                        "CV_PR_AUC",
                        "CV_Threshold_Precision",
                    ],
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            # ----------------------------------------------------
            # Select best model
            # ----------------------------------------------------

            best_model_name = (
                results_df.iloc[0]["Model"]
            )

            best_model = (
                trained_models[
                    best_model_name
                ]["model"]
            )

            best_threshold = (
                trained_models[
                    best_model_name
                ]["threshold"]
            )

            # ----------------------------------------------------
            # Save artifacts
            # ----------------------------------------------------

            self._save_artifacts(
                best_model=best_model,
                best_model_name=best_model_name,
                best_threshold=best_threshold,
                results_df=results_df,
            )

            # ----------------------------------------------------
            # Display results
            # ----------------------------------------------------

            print(
                "\n"
                + "=" * 120
            )

            print(
                "MODEL TRAINING RESULTS"
            )

            print(
                "=" * 120
            )

            print(
                results_df.to_string(
                    index=False
                )
            )

            print(
                "\n"
                + "=" * 120
            )

            print(
                f"Selected Model: {best_model_name}"
            )

            print(
                f"Selected Threshold: "
                f"{best_threshold:.3f}"
            )

            print(
                "Model selection was based on "
                "cross-validation only."
            )

            print(
                "The final test set was NOT used "
                "for model selection."
            )

            print(
                "=" * 120
            )

            logger.info(
                "Best model: %s",
                best_model_name,
            )

            logger.info(
                "Best threshold: %.3f",
                best_threshold,
            )

            logger.info(
                "Model training completed successfully."
            )

            return (
                best_model,
                results_df,
                best_threshold,
            )

        except Exception as exc:

            logger.exception(
                "Model training failed."
            )

            raise CustomException(
                exc,
                sys,
            ) from exc


# ================================================================
# MODULE TEST
# ================================================================

if __name__ == "__main__":

    print(
        "ModelTraining module loaded successfully."
    )