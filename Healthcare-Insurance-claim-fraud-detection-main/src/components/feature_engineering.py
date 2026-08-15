from __future__ import annotations

import sys
from typing import Tuple

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import get_logger


logger = get_logger(__name__)


class FeatureEngineering:
    """
    Builds provider-level behavioral features for healthcare
    insurance fraud detection.

    Input:
        - provider_df
        - beneficiary_df
        - inpatient_df
        - outpatient_df

    Output:
        One row per provider containing numerical behavioral
        features and the PotentialFraud target.
    """

    def __init__(self) -> None:
        logger.info("FeatureEngineering initialized.")

    # ============================================================
    # GENERAL HELPERS
    # ============================================================

    @staticmethod
    def _safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series:
        """
        Performs safe division and avoids division-by-zero errors.
        """

        result = numerator.div(
            denominator.replace(0, np.nan)
        )

        return result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    @staticmethod
    def _to_numeric(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        """
        Converts available columns to numeric values.
        """

        for column in columns:

            if column in dataframe.columns:

                dataframe[column] = pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )

        return dataframe

    @staticmethod
    def _count_non_null(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> pd.Series:
        """
        Counts how many non-null diagnosis/procedure values
        exist in each claim.
        """

        existing_columns = [
            column
            for column in columns
            if column in dataframe.columns
        ]

        if not existing_columns:

            return pd.Series(
                0,
                index=dataframe.index,
                dtype=float,
            )

        return dataframe[
            existing_columns
        ].notna().sum(axis=1)

    # ============================================================
    # DATE PREPARATION
    # ============================================================

    @staticmethod
    def _prepare_dates(
        beneficiary_df: pd.DataFrame,
        inpatient_df: pd.DataFrame,
        outpatient_df: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Converts all relevant date columns into datetime.
        """

        beneficiary = beneficiary_df.copy()
        inpatient = inpatient_df.copy()
        outpatient = outpatient_df.copy()

        # --------------------------------------------------------
        # Beneficiary dates
        # --------------------------------------------------------

        for column in [
            "DOB",
            "DOD",
        ]:

            if column in beneficiary.columns:

                beneficiary[column] = pd.to_datetime(
                    beneficiary[column],
                    errors="coerce",
                )

        # --------------------------------------------------------
        # Inpatient dates
        # --------------------------------------------------------

        for column in [
            "ClaimStartDt",
            "ClaimEndDt",
            "AdmissionDt",
            "DischargeDt",
        ]:

            if column in inpatient.columns:

                inpatient[column] = pd.to_datetime(
                    inpatient[column],
                    errors="coerce",
                )

        # --------------------------------------------------------
        # Outpatient dates
        # --------------------------------------------------------

        for column in [
            "ClaimStartDt",
            "ClaimEndDt",
        ]:

            if column in outpatient.columns:

                outpatient[column] = pd.to_datetime(
                    outpatient[column],
                    errors="coerce",
                )

        return (
            beneficiary,
            inpatient,
            outpatient,
        )

    # ============================================================
    # INPATIENT CLAIM FEATURES
    # ============================================================

    def _create_inpatient_features(
        self,
        inpatient_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates claim-level features for inpatient claims.
        """

        df = inpatient_df.copy()

        logger.info(
            "Creating inpatient claim-level features."
        )

        # --------------------------------------------------------
        # Claim duration
        # --------------------------------------------------------

        if {
            "ClaimStartDt",
            "ClaimEndDt",
        }.issubset(df.columns):

            df["ClaimDuration"] = (
                df["ClaimEndDt"]
                - df["ClaimStartDt"]
            ).dt.days

        else:

            df["ClaimDuration"] = np.nan

        # --------------------------------------------------------
        # Hospital stay
        # --------------------------------------------------------

        if {
            "AdmissionDt",
            "DischargeDt",
        }.issubset(df.columns):

            df["HospitalStayDays"] = (
                df["DischargeDt"]
                - df["AdmissionDt"]
            ).dt.days

        else:

            df["HospitalStayDays"] = np.nan

        # Remove impossible negative values.
        df["ClaimDuration"] = df[
            "ClaimDuration"
        ].where(
            df["ClaimDuration"] >= 0
        )

        df["HospitalStayDays"] = df[
            "HospitalStayDays"
        ].where(
            df["HospitalStayDays"] >= 0
        )

        # --------------------------------------------------------
        # Claim amount
        # --------------------------------------------------------

        if "InscClaimAmtReimbursed" in df.columns:

            df["ClaimAmount"] = pd.to_numeric(
                df["InscClaimAmtReimbursed"],
                errors="coerce",
            )

        else:

            df["ClaimAmount"] = np.nan

        # --------------------------------------------------------
        # Deductible
        # --------------------------------------------------------

        if "DeductibleAmtPaid" in df.columns:

            df["DeductibleAmount"] = pd.to_numeric(
                df["DeductibleAmtPaid"],
                errors="coerce",
            )

        else:

            df["DeductibleAmount"] = np.nan

        # --------------------------------------------------------
        # Long hospital stay
        # --------------------------------------------------------

        df["LongHospitalStay"] = (
            df["HospitalStayDays"] > 7
        ).astype(int)

        # --------------------------------------------------------
        # Diagnosis count
        # --------------------------------------------------------

        diagnosis_columns = [
            column
            for column in df.columns
            if column.startswith(
                "ClmDiagnosisCode_"
            )
        ]

        df["DiagnosisCount"] = (
            self._count_non_null(
                df,
                diagnosis_columns,
            )
        )

        # --------------------------------------------------------
        # Procedure count
        # --------------------------------------------------------

        procedure_columns = [
            column
            for column in df.columns
            if column.startswith(
                "ClmProcedureCode_"
            )
        ]

        df["ProcedureCount"] = (
            self._count_non_null(
                df,
                procedure_columns,
            )
        )

        return df

    # ============================================================
    # OUTPATIENT CLAIM FEATURES
    # ============================================================

    def _create_outpatient_features(
        self,
        outpatient_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates claim-level features for outpatient claims.
        """

        df = outpatient_df.copy()

        logger.info(
            "Creating outpatient claim-level features."
        )

        # --------------------------------------------------------
        # Claim duration
        # --------------------------------------------------------

        if {
            "ClaimStartDt",
            "ClaimEndDt",
        }.issubset(df.columns):

            df["ClaimDuration"] = (
                df["ClaimEndDt"]
                - df["ClaimStartDt"]
            ).dt.days

        else:

            df["ClaimDuration"] = np.nan

        df["ClaimDuration"] = df[
            "ClaimDuration"
        ].where(
            df["ClaimDuration"] >= 0
        )

        # --------------------------------------------------------
        # Claim amount
        # --------------------------------------------------------

        if "InscClaimAmtReimbursed" in df.columns:

            df["ClaimAmount"] = pd.to_numeric(
                df["InscClaimAmtReimbursed"],
                errors="coerce",
            )

        else:

            df["ClaimAmount"] = np.nan

        # --------------------------------------------------------
        # Deductible
        # --------------------------------------------------------

        if "DeductibleAmtPaid" in df.columns:

            df["DeductibleAmount"] = pd.to_numeric(
                df["DeductibleAmtPaid"],
                errors="coerce",
            )

        else:

            df["DeductibleAmount"] = np.nan

        # --------------------------------------------------------
        # Diagnosis count
        # --------------------------------------------------------

        diagnosis_columns = [
            column
            for column in df.columns
            if column.startswith(
                "ClmDiagnosisCode_"
            )
        ]

        df["DiagnosisCount"] = (
            self._count_non_null(
                df,
                diagnosis_columns,
            )
        )

        # --------------------------------------------------------
        # Procedure count
        # --------------------------------------------------------

        procedure_columns = [
            column
            for column in df.columns
            if column.startswith(
                "ClmProcedureCode_"
            )
        ]

        df["ProcedureCount"] = (
            self._count_non_null(
                df,
                procedure_columns,
            )
        )

        return df

    # ============================================================
    # BENEFICIARY FEATURES
    # ============================================================

    def _create_beneficiary_features(
        self,
        beneficiary_df: pd.DataFrame,
        inpatient_df: pd.DataFrame,
        outpatient_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates beneficiary-level features.

        Age is calculated using the beneficiary DOB and the
        first claim date associated with that beneficiary.
        """

        beneficiary = beneficiary_df.copy()

        logger.info(
            "Creating beneficiary-level features."
        )

        # --------------------------------------------------------
        # Chronic conditions
        # --------------------------------------------------------

        chronic_columns = [
            column
            for column in beneficiary.columns
            if column.startswith(
                "ChronicCond_"
            )
        ]

        if chronic_columns:

            chronic_data = beneficiary[
                chronic_columns
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )

            beneficiary[
                "ChronicConditionCount"
            ] = (
                chronic_data.eq(1)
                .sum(axis=1)
            )

        else:

            beneficiary[
                "ChronicConditionCount"
            ] = 0

        # --------------------------------------------------------
        # Numeric beneficiary fields
        # --------------------------------------------------------

        beneficiary = self._to_numeric(
            beneficiary,
            [
                "Gender",
                "State",
                "County",
                "NoOfMonths_PartACov",
                "NoOfMonths_PartBCov",
                "IPAnnualReimbursementAmt",
                "IPAnnualDeductibleAmt",
                "OPAnnualReimbursementAmt",
                "OPAnnualDeductibleAmt",
            ],
        )

        # --------------------------------------------------------
        # Renal disease
        # --------------------------------------------------------

        if "RenalDiseaseIndicator" in beneficiary.columns:

            beneficiary[
                "RenalDiseaseFlag"
            ] = (
                beneficiary[
                    "RenalDiseaseIndicator"
                ]
                .astype(str)
                .str.upper()
                .eq("Y")
                .astype(int)
            )

        else:

            beneficiary[
                "RenalDiseaseFlag"
            ] = 0

        # --------------------------------------------------------
        # Find first claim date for each beneficiary
        # --------------------------------------------------------

        claim_date_frames = []

        if not inpatient_df.empty:

            if {
                "BeneID",
                "ClaimStartDt",
            }.issubset(
                inpatient_df.columns
            ):

                claim_date_frames.append(
                    inpatient_df[
                        [
                            "BeneID",
                            "ClaimStartDt",
                        ]
                    ]
                )

        if not outpatient_df.empty:

            if {
                "BeneID",
                "ClaimStartDt",
            }.issubset(
                outpatient_df.columns
            ):

                claim_date_frames.append(
                    outpatient_df[
                        [
                            "BeneID",
                            "ClaimStartDt",
                        ]
                    ]
                )

        if (
            "DOB" in beneficiary.columns
            and claim_date_frames
        ):

            all_claim_dates = pd.concat(
                claim_date_frames,
                ignore_index=True,
            )

            first_claim_dates = (
                all_claim_dates
                .dropna(
                    subset=[
                        "BeneID",
                        "ClaimStartDt",
                    ]
                )
                .groupby("BeneID")[
                    "ClaimStartDt"
                ]
                .min()
            )

            beneficiary = beneficiary.merge(
                first_claim_dates.rename(
                    "FirstClaimDate"
                ),
                left_on="BeneID",
                right_index=True,
                how="left",
            )

            beneficiary[
                "BeneficiaryAge"
            ] = (
                (
                    beneficiary[
                        "FirstClaimDate"
                    ]
                    - beneficiary["DOB"]
                ).dt.days
                / 365.25
            )

            beneficiary[
                "BeneficiaryAge"
            ] = beneficiary[
                "BeneficiaryAge"
            ].clip(
                lower=0,
                upper=120,
            )

        else:

            beneficiary[
                "BeneficiaryAge"
            ] = np.nan

        return beneficiary

    # ============================================================
    # CLAIM AGGREGATION
    # ============================================================

    def _aggregate_claim_features(
        self,
        inpatient_df: pd.DataFrame,
        outpatient_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregates claim behavior to provider level.
        """

        logger.info(
            "Aggregating claims at provider level."
        )

        provider_frames = []

        # ========================================================
        # INPATIENT
        # ========================================================

        if not inpatient_df.empty:

            inpatient = (
                inpatient_df
                .groupby("Provider")
                .agg(
                    InpatientClaims=(
                        "ClaimID",
                        "count",
                    ),

                    InpatientReimbursement=(
                        "ClaimAmount",
                        "sum",
                    ),

                    InpatientAverageClaim=(
                        "ClaimAmount",
                        "mean",
                    ),

                    InpatientMaximumClaim=(
                        "ClaimAmount",
                        "max",
                    ),

                    AverageHospitalStay=(
                        "HospitalStayDays",
                        "mean",
                    ),

                    MaximumHospitalStay=(
                        "HospitalStayDays",
                        "max",
                    ),

                    AverageInpatientClaimDuration=(
                        "ClaimDuration",
                        "mean",
                    ),

                    MaximumInpatientClaimDuration=(
                        "ClaimDuration",
                        "max",
                    ),

                    AverageInpatientDeductible=(
                        "DeductibleAmount",
                        "mean",
                    ),

                    TotalInpatientDeductible=(
                        "DeductibleAmount",
                        "sum",
                    ),

                    LongHospitalStayCount=(
                        "LongHospitalStay",
                        "sum",
                    ),
                )
            )

            inpatient[
                "LongHospitalStayRatio"
            ] = self._safe_divide(
                inpatient[
                    "LongHospitalStayCount"
                ],
                inpatient[
                    "InpatientClaims"
                ],
            )

            # ----------------------------------------------------
            # High-value inpatient claims
            # ----------------------------------------------------

            threshold = (
                inpatient_df[
                    "ClaimAmount"
                ].quantile(0.95)
            )

            inpatient_high_value = (
                inpatient_df
                .assign(
                    HighValue=(
                        inpatient_df[
                            "ClaimAmount"
                        ] > threshold
                    ).astype(int)
                )
                .groupby("Provider")[
                    "HighValue"
                ]
                .mean()
                .rename(
                    "HighValueInpatientClaimRatio"
                )
            )

            inpatient = inpatient.join(
                inpatient_high_value
            )

            # ----------------------------------------------------
            # Long claim duration
            # ----------------------------------------------------

            inpatient_long_duration = (
                inpatient_df
                .assign(
                    LongDuration=(
                        inpatient_df[
                            "ClaimDuration"
                        ] > 7
                    ).astype(int)
                )
                .groupby("Provider")[
                    "LongDuration"
                ]
                .mean()
                .rename(
                    "LongClaimDurationRatio"
                )
            )

            inpatient = inpatient.join(
                inpatient_long_duration
            )

            provider_frames.append(
                inpatient
            )

        # ========================================================
        # OUTPATIENT
        # ========================================================

        if not outpatient_df.empty:

            outpatient = (
                outpatient_df
                .groupby("Provider")
                .agg(
                    OutpatientClaims=(
                        "ClaimID",
                        "count",
                    ),

                    OutpatientReimbursement=(
                        "ClaimAmount",
                        "sum",
                    ),

                    OutpatientAverageClaim=(
                        "ClaimAmount",
                        "mean",
                    ),

                    OutpatientMaximumClaim=(
                        "ClaimAmount",
                        "max",
                    ),

                    AverageOutpatientClaimDuration=(
                        "ClaimDuration",
                        "mean",
                    ),

                    MaximumOutpatientClaimDuration=(
                        "ClaimDuration",
                        "max",
                    ),

                    AverageOutpatientDeductible=(
                        "DeductibleAmount",
                        "mean",
                    ),

                    TotalOutpatientDeductible=(
                        "DeductibleAmount",
                        "sum",
                    ),
                )
            )

            # ----------------------------------------------------
            # High-value outpatient claims
            # ----------------------------------------------------

            threshold = (
                outpatient_df[
                    "ClaimAmount"
                ].quantile(0.95)
            )

            outpatient_high_value = (
                outpatient_df
                .assign(
                    HighValue=(
                        outpatient_df[
                            "ClaimAmount"
                        ] > threshold
                    ).astype(int)
                )
                .groupby("Provider")[
                    "HighValue"
                ]
                .mean()
                .rename(
                    "HighValueOutpatientClaimRatio"
                )
            )

            outpatient = outpatient.join(
                outpatient_high_value
            )

            # ----------------------------------------------------
            # Long outpatient claim duration
            # ----------------------------------------------------

            outpatient_long_duration = (
                outpatient_df
                .assign(
                    LongDuration=(
                        outpatient_df[
                            "ClaimDuration"
                        ] > 7
                    ).astype(int)
                )
                .groupby("Provider")[
                    "LongDuration"
                ]
                .mean()
                .rename(
                    "LongOutpatientClaimDurationRatio"
                )
            )

            outpatient = outpatient.join(
                outpatient_long_duration
            )

            provider_frames.append(
                outpatient
            )

        # ========================================================
        # COMBINE
        # ========================================================

        if not provider_frames:

            return pd.DataFrame(
                columns=[
                    "Provider"
                ]
            )

        result = pd.concat(
            provider_frames,
            axis=1,
        )

        result = (
            result.loc[
                :,
                ~result.columns.duplicated(),
            ]
        )

        return result.reset_index()

    # ============================================================
    # CLAIM DISTRIBUTION FEATURES
    # ============================================================

    def _create_claim_distribution_features(
        self,
        inpatient_df: pd.DataFrame,
        outpatient_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates distribution-based claim features.

        These are useful because fraud can appear as unusual
        claim-value or claim-duration behavior.
        """

        logger.info(
            "Creating claim distribution features."
        )

        claim_frames = []

        if not inpatient_df.empty:

            inpatient_claims = (
                inpatient_df[
                    [
                        "Provider",
                        "ClaimAmount",
                        "DeductibleAmount",
                        "ClaimDuration",
                    ]
                ]
                .copy()
            )

            claim_frames.append(
                inpatient_claims
            )

        if not outpatient_df.empty:

            outpatient_claims = (
                outpatient_df[
                    [
                        "Provider",
                        "ClaimAmount",
                        "DeductibleAmount",
                        "ClaimDuration",
                    ]
                ]
                .copy()
            )

            claim_frames.append(
                outpatient_claims
            )

        if not claim_frames:

            return pd.DataFrame(
                columns=[
                    "Provider"
                ]
            )

        claims = pd.concat(
            claim_frames,
            ignore_index=True,
        )

        # --------------------------------------------------------
        # Claim distribution
        # --------------------------------------------------------

        distribution = (
            claims
            .groupby("Provider")
            .agg(
                MedianClaimAmount=(
                    "ClaimAmount",
                    "median",
                ),

                ClaimAmountStd=(
                    "ClaimAmount",
                    "std",
                ),

                MaximumClaimAmount=(
                    "ClaimAmount",
                    "max",
                ),

                AverageClaimDuration=(
                    "ClaimDuration",
                    "mean",
                ),

                MaximumClaimDuration=(
                    "ClaimDuration",
                    "max",
                ),

                AverageDeductiblePaid=(
                    "DeductibleAmount",
                    "mean",
                ),
            )
        )

        # --------------------------------------------------------
        # Reimbursement per claim
        # --------------------------------------------------------

        distribution[
            "ReimbursementPerClaim"
        ] = (
            claims
            .groupby("Provider")[
                "ClaimAmount"
            ]
            .sum()
            .div(
                claims
                .groupby("Provider")
                .size()
            )
        )

        # --------------------------------------------------------
        # Deductible per claim
        # --------------------------------------------------------

        distribution[
            "DeductiblePerClaim"
        ] = (
            claims
            .groupby("Provider")[
                "DeductibleAmount"
            ]
            .sum()
            .div(
                claims
                .groupby("Provider")
                .size()
            )
        )

        # --------------------------------------------------------
        # Deductible / reimbursement
        # --------------------------------------------------------

        total_deductible = (
            claims
            .groupby("Provider")[
                "DeductibleAmount"
            ]
            .sum()
        )

        total_reimbursement = (
            claims
            .groupby("Provider")[
                "ClaimAmount"
            ]
            .sum()
        )

        distribution[
            "DeductibleToReimbursementRatio"
        ] = self._safe_divide(
            total_deductible,
            total_reimbursement,
        )

        return distribution.reset_index()

    # ============================================================
    # BENEFICIARY AGGREGATION
    # ============================================================

    def _aggregate_beneficiary_features(
        self,
        beneficiary_df: pd.DataFrame,
        inpatient_df: pd.DataFrame,
        outpatient_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates provider-level beneficiary behavior features.
        """

        logger.info(
            "Aggregating beneficiary behavior by provider."
        )

        claim_frames = []

        # --------------------------------------------------------
        # Inpatient claims
        # --------------------------------------------------------

        if not inpatient_df.empty:

            claim_frames.append(
                inpatient_df[
                    [
                        "Provider",
                        "BeneID",
                        "ClaimAmount",
                    ]
                ].copy()
            )

        # --------------------------------------------------------
        # Outpatient claims
        # --------------------------------------------------------

        if not outpatient_df.empty:

            claim_frames.append(
                outpatient_df[
                    [
                        "Provider",
                        "BeneID",
                        "ClaimAmount",
                    ]
                ].copy()
            )

        if not claim_frames:

            return pd.DataFrame(
                columns=[
                    "Provider"
                ]
            )

        claims = pd.concat(
            claim_frames,
            ignore_index=True,
        )

        # --------------------------------------------------------
        # Claims per beneficiary
        # --------------------------------------------------------

        beneficiary_claim_counts = (
            claims
            .groupby(
                [
                    "Provider",
                    "BeneID",
                ]
            )
            .size()
            .rename(
                "BeneficiaryClaimCount"
            )
            .reset_index()
        )

        provider_beneficiary = (
            beneficiary_claim_counts
            .groupby("Provider")
            .agg(
                UniqueBeneficiaries=(
                    "BeneID",
                    "nunique",
                ),

                MaximumBeneficiaryClaims=(
                    "BeneficiaryClaimCount",
                    "max",
                ),

                AverageBeneficiaryClaims=(
                    "BeneficiaryClaimCount",
                    "mean",
                ),
            )
        )

        provider_beneficiary[
            "ClaimsPerBeneficiary"
        ] = self._safe_divide(
            claims
            .groupby("Provider")
            .size(),
            provider_beneficiary[
                "UniqueBeneficiaries"
            ],
        )

        # --------------------------------------------------------
        # Beneficiary concentration
        # --------------------------------------------------------

        total_claims = (
            beneficiary_claim_counts
            .groupby("Provider")[
                "BeneficiaryClaimCount"
            ]
            .sum()
        )

        maximum_claims = (
            beneficiary_claim_counts
            .groupby("Provider")[
                "BeneficiaryClaimCount"
            ]
            .max()
        )

        provider_beneficiary[
            "BeneficiaryConcentration"
        ] = self._safe_divide(
            maximum_claims,
            total_claims,
        )

        # --------------------------------------------------------
        # Beneficiary information
        # --------------------------------------------------------

        beneficiary_columns = [
            "BeneID",
            "BeneficiaryAge",
            "ChronicConditionCount",
            "Gender",
            "NoOfMonths_PartACov",
            "NoOfMonths_PartBCov",
            "IPAnnualReimbursementAmt",
            "OPAnnualReimbursementAmt",
            "IPAnnualDeductibleAmt",
            "OPAnnualDeductibleAmt",
            "RenalDiseaseFlag",
            "State",
            "County",
        ]

        available_columns = [
            column
            for column in beneficiary_columns
            if column in beneficiary_df.columns
        ]

        beneficiary_info = (
            beneficiary_df[
                available_columns
            ]
            .drop_duplicates(
                subset=["BeneID"]
            )
        )

        claims_with_beneficiary = (
            claims.merge(
                beneficiary_info,
                on="BeneID",
                how="left",
            )
        )

        # --------------------------------------------------------
        # Provider demographic behavior
        # --------------------------------------------------------

        demographic_features = (
            claims_with_beneficiary
            .groupby("Provider")
            .agg(
                AverageChronicConditions=(
                    "ChronicConditionCount",
                    "mean",
                ),

                MaximumChronicConditions=(
                    "ChronicConditionCount",
                    "max",
                ),

                AverageGender=(
                    "Gender",
                    "mean",
                ),

                AveragePartACoverageMonths=(
                    "NoOfMonths_PartACov",
                    "mean",
                ),

                AveragePartBCoverageMonths=(
                    "NoOfMonths_PartBCov",
                    "mean",
                ),

                AverageBeneficiaryIPReimbursement=(
                    "IPAnnualReimbursementAmt",
                    "mean",
                ),

                AverageBeneficiaryOPReimbursement=(
                    "OPAnnualReimbursementAmt",
                    "mean",
                ),

                AverageBeneficiaryIPDeductible=(
                    "IPAnnualDeductibleAmt",
                    "mean",
                ),

                AverageBeneficiaryOPDeductible=(
                    "OPAnnualDeductibleAmt",
                    "mean",
                ),

                AverageRenalDiseaseIndicator=(
                    "RenalDiseaseFlag",
                    "mean",
                ),

                UniqueBeneficiaryStates=(
                    "State",
                    "nunique",
                ),

                UniqueBeneficiaryCounties=(
                    "County",
                    "nunique",
                ),

                AverageBeneficiaryAge=(
                    "BeneficiaryAge",
                    "mean",
                ),

                MedianBeneficiaryAge=(
                    "BeneficiaryAge",
                    "median",
                ),

                MinimumBeneficiaryAge=(
                    "BeneficiaryAge",
                    "min",
                ),

                MaximumBeneficiaryAge=(
                    "BeneficiaryAge",
                    "max",
                ),
            )
        )

        # --------------------------------------------------------
        # Senior beneficiary ratio
        # --------------------------------------------------------

        senior_ratio = (
            claims_with_beneficiary
            .assign(
                Senior=(
                    claims_with_beneficiary[
                        "BeneficiaryAge"
                    ] >= 65
                ).astype(int)
            )
            .groupby("Provider")[
                "Senior"
            ]
            .mean()
            .rename(
                "SeniorBeneficiaryRatio"
            )
        )

        demographic_features = (
            demographic_features.join(
                senior_ratio
            )
        )

        # --------------------------------------------------------
        # Female beneficiary ratio
        # --------------------------------------------------------

        female_ratio = (
            claims_with_beneficiary
            .assign(
                Female=(
                    claims_with_beneficiary[
                        "Gender"
                    ] == 2
                ).astype(int)
            )
            .groupby("Provider")[
                "Female"
            ]
            .mean()
            .rename(
                "FemaleBeneficiaryRatio"
            )
        )

        demographic_features = (
            demographic_features.join(
                female_ratio
            )
        )

        provider_beneficiary = (
            provider_beneficiary.join(
                demographic_features,
                how="left",
            )
        )

        return provider_beneficiary.reset_index()

    # ============================================================
    # PHYSICIAN FEATURES
    # ============================================================

    def _aggregate_physician_features(
        self,
        inpatient_df: pd.DataFrame,
        outpatient_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Converts physician identifiers into provider behavior
        rather than feeding raw physician IDs to the model.
        """

        logger.info(
            "Creating physician behavior features."
        )

        claim_frames = []

        for dataframe in [
            inpatient_df,
            outpatient_df,
        ]:

            if dataframe.empty:
                continue

            columns = [
                "Provider",
                "ClaimID",
                "AttendingPhysician",
                "OperatingPhysician",
                "OtherPhysician",
            ]

            available = [
                column
                for column in columns
                if column in dataframe.columns
            ]

            claim_frames.append(
                dataframe[
                    available
                ].copy()
            )

        if not claim_frames:

            return pd.DataFrame(
                columns=[
                    "Provider"
                ]
            )

        claims = pd.concat(
            claim_frames,
            ignore_index=True,
        )

        physician_features = (
            claims
            .groupby("Provider")
            .agg(
                UniqueAttendingPhysicians=(
                    "AttendingPhysician",
                    "nunique",
                ),

                UniqueOperatingPhysicians=(
                    "OperatingPhysician",
                    "nunique",
                ),

                UniqueOtherPhysicians=(
                    "OtherPhysician",
                    "nunique",
                ),

                ProviderClaimCount=(
                    "ClaimID",
                    "count",
                ),
            )
        )

        physician_features[
            "ClaimsPerAttendingPhysician"
        ] = self._safe_divide(
            physician_features[
                "ProviderClaimCount"
            ],
            physician_features[
                "UniqueAttendingPhysicians"
            ],
        )

        physician_features[
            "ClaimsPerOperatingPhysician"
        ] = self._safe_divide(
            physician_features[
                "ProviderClaimCount"
            ],
            physician_features[
                "UniqueOperatingPhysicians"
            ],
        )

        physician_features[
            "ClaimsPerOtherPhysician"
        ] = self._safe_divide(
            physician_features[
                "ProviderClaimCount"
            ],
            physician_features[
                "UniqueOtherPhysicians"
            ],
        )

        physician_features = (
            physician_features.drop(
                columns=[
                    "ProviderClaimCount"
                ]
            )
        )

        return physician_features.reset_index()

    # ============================================================
    # DIAGNOSIS / PROCEDURE FEATURES
    # ============================================================

    def _aggregate_diagnosis_procedure_features(
        self,
        inpatient_df: pd.DataFrame,
        outpatient_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates provider-level diagnosis and procedure diversity.
        """

        logger.info(
            "Creating diagnosis and procedure diversity features."
        )

        diagnosis_frames = []
        procedure_frames = []

        for dataframe in [
            inpatient_df,
            outpatient_df,
        ]:

            if dataframe.empty:
                continue

            # ----------------------------------------------------
            # Diagnosis columns
            # ----------------------------------------------------

            diagnosis_columns = [
                column
                for column in dataframe.columns
                if column.startswith(
                    "ClmDiagnosisCode_"
                )
            ]

            if diagnosis_columns:

                diagnosis_data = (
                    dataframe[
                        [
                            "Provider"
                        ]
                        + diagnosis_columns
                    ]
                    .melt(
                        id_vars=[
                            "Provider"
                        ],
                        value_vars=diagnosis_columns,
                        value_name="DiagnosisCode",
                    )
                )

                diagnosis_frames.append(
                    diagnosis_data[
                        [
                            "Provider",
                            "DiagnosisCode",
                        ]
                    ]
                )

            # ----------------------------------------------------
            # Procedure columns
            # ----------------------------------------------------

            procedure_columns = [
                column
                for column in dataframe.columns
                if column.startswith(
                    "ClmProcedureCode_"
                )
            ]

            if procedure_columns:

                procedure_data = (
                    dataframe[
                        [
                            "Provider"
                        ]
                        + procedure_columns
                    ]
                    .melt(
                        id_vars=[
                            "Provider"
                        ],
                        value_vars=procedure_columns,
                        value_name="ProcedureCode",
                    )
                )

                procedure_frames.append(
                    procedure_data[
                        [
                            "Provider",
                            "ProcedureCode",
                        ]
                    ]
                )

        result = pd.DataFrame()

        # --------------------------------------------------------
        # Diagnosis diversity
        # --------------------------------------------------------

        if diagnosis_frames:

            diagnosis = pd.concat(
                diagnosis_frames,
                ignore_index=True,
            )

            diagnosis = diagnosis.dropna(
                subset=[
                    "DiagnosisCode"
                ]
            )

            diagnosis_counts = (
                diagnosis
                .groupby("Provider")[
                    "DiagnosisCode"
                ]
                .nunique()
                .rename(
                    "UniqueDiagnosisCount"
                )
            )

            result = pd.DataFrame(
                diagnosis_counts
            )

        # --------------------------------------------------------
        # Procedure diversity
        # --------------------------------------------------------

        if procedure_frames:

            procedure = pd.concat(
                procedure_frames,
                ignore_index=True,
            )

            procedure = procedure.dropna(
                subset=[
                    "ProcedureCode"
                ]
            )

            procedure_counts = (
                procedure
                .groupby("Provider")[
                    "ProcedureCode"
                ]
                .nunique()
                .rename(
                    "UniqueProcedureCount"
                )
            )

            if result.empty:

                result = pd.DataFrame(
                    procedure_counts
                )

            else:

                result = result.join(
                    procedure_counts,
                    how="outer",
                )

        # --------------------------------------------------------
        # Claims per diagnosis / procedure
        # --------------------------------------------------------

        claim_counts = []

        for dataframe in [
            inpatient_df,
            outpatient_df,
        ]:

            if not dataframe.empty:

                claim_counts.append(
                    dataframe
                    .groupby("Provider")
                    .size()
                )

        if claim_counts and not result.empty:

            total_claims = (
                pd.concat(
                    claim_counts,
                    axis=1,
                )
                .fillna(0)
                .sum(axis=1)
            )

            result = result.join(
                total_claims.rename(
                    "TotalClaimsForCodes"
                ),
                how="left",
            )

            if "UniqueDiagnosisCount" in result:

                result[
                    "DiagnosesPerClaim"
                ] = self._safe_divide(
                    result[
                        "UniqueDiagnosisCount"
                    ],
                    result[
                        "TotalClaimsForCodes"
                    ],
                )

            if "UniqueProcedureCount" in result:

                result[
                    "ProceduresPerClaim"
                ] = self._safe_divide(
                    result[
                        "UniqueProcedureCount"
                    ],
                    result[
                        "TotalClaimsForCodes"
                    ],
                )

            result = result.drop(
                columns=[
                    "TotalClaimsForCodes"
                ]
            )

        return result.reset_index()


    # ============================================================
    # ADVANCED BEHAVIORAL FEATURES
    # ============================================================

    def _create_advanced_behavior_features(
        self,
        inpatient_df: pd.DataFrame,
        outpatient_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates additional provider-level behavioral features.

        Focus:
        - claim amount distribution and outliers
        - financial intensity
        - beneficiary concentration
        - monthly claim activity
        - provider activity span

        PotentialFraud is never used here.
        """

        logger.info("Creating advanced behavioral features.")

        frames = []

        for dataframe in [inpatient_df, outpatient_df]:
            if dataframe.empty:
                continue

            columns = [
                "Provider",
                "ClaimID",
                "BeneID",
                "ClaimAmount",
                "DeductibleAmount",
                "ClaimDuration",
                "ClaimStartDt",
            ]

            available = [
                column for column in columns
                if column in dataframe.columns
            ]

            frames.append(dataframe[available].copy())

        if not frames:
            return pd.DataFrame(columns=["Provider"])

        claims = pd.concat(frames, ignore_index=True)

        for column in [
            "ClaimAmount",
            "DeductibleAmount",
            "ClaimDuration",
        ]:
            if column in claims.columns:
                claims[column] = pd.to_numeric(
                    claims[column],
                    errors="coerce",
                )

        grouped = claims.groupby("Provider")

        result = grouped.agg(
            ClaimAmountQ25=(
                "ClaimAmount",
                lambda x: x.quantile(0.25),
            ),
            ClaimAmountQ75=(
                "ClaimAmount",
                lambda x: x.quantile(0.75),
            ),
            ClaimAmountMin=(
                "ClaimAmount",
                "min",
            ),
            ClaimAmountMax=(
                "ClaimAmount",
                "max",
            ),
            ClaimAmountMeanAdvanced=(
                "ClaimAmount",
                "mean",
            ),
            ClaimAmountMedianAdvanced=(
                "ClaimAmount",
                "median",
            ),
            ClaimAmountStdAdvanced=(
                "ClaimAmount",
                "std",
            ),
            ClaimDurationMedianAdvanced=(
                "ClaimDuration",
                "median",
            ),
            ClaimDurationStdAdvanced=(
                "ClaimDuration",
                "std",
            ),
        )

        result["ClaimAmountIQR"] = (
            result["ClaimAmountQ75"]
            - result["ClaimAmountQ25"]
        )

        result["ClaimAmountRange"] = (
            result["ClaimAmountMax"]
            - result["ClaimAmountMin"]
        )

        result["ClaimAmountMeanMedianRatio"] = (
            self._safe_divide(
                result["ClaimAmountMeanAdvanced"],
                result["ClaimAmountMedianAdvanced"],
            )
        )

        result["ClaimAmountOutlierRatio"] = (
            self._safe_divide(
                result["ClaimAmountMax"],
                result["ClaimAmountMedianAdvanced"],
            )
        )

        result["ClaimAmountCV"] = (
            self._safe_divide(
                result["ClaimAmountStdAdvanced"],
                result["ClaimAmountMeanAdvanced"],
            )
        )

        result["LogClaimAmountCV"] = np.log1p(
            result["ClaimAmountCV"].clip(lower=0)
        )

        result["ClaimDurationCV"] = (
            self._safe_divide(
                result["ClaimDurationStdAdvanced"],
                result["ClaimDurationMedianAdvanced"],
            )
        )

        total_claims = grouped.size()

        total_reimbursement = grouped[
            "ClaimAmount"
        ].sum()

        total_deductible = grouped[
            "DeductibleAmount"
        ].sum()

        result["ReimbursementPerClaimAdvanced"] = (
            self._safe_divide(
                total_reimbursement,
                total_claims,
            )
        )

        result["DeductiblePerClaimAdvanced"] = (
            self._safe_divide(
                total_deductible,
                total_claims,
            )
        )

        result["DeductibleToReimbursementAdvanced"] = (
            self._safe_divide(
                total_deductible,
                total_reimbursement,
            )
        )

        # --------------------------------------------------------
        # Beneficiary concentration
        # --------------------------------------------------------

        if "BeneID" in claims.columns:

            beneficiary_counts = (
                claims.groupby(
                    ["Provider", "BeneID"]
                )
                .size()
                .rename("BeneficiaryClaimCount")
                .reset_index()
            )

            beneficiary_summary = (
                beneficiary_counts
                .groupby("Provider")
                .agg(
                    BeneficiaryClaimMedian=(
                        "BeneficiaryClaimCount",
                        "median",
                    ),
                    BeneficiaryClaimStd=(
                        "BeneficiaryClaimCount",
                        "std",
                    ),
                    MaximumBeneficiaryClaimsAdvanced=(
                        "BeneficiaryClaimCount",
                        "max",
                    ),
                )
            )

            beneficiary_summary[
                "TopBeneficiaryClaimShare"
            ] = self._safe_divide(
                beneficiary_counts
                .groupby("Provider")[
                    "BeneficiaryClaimCount"
                ]
                .max(),
                beneficiary_counts
                .groupby("Provider")[
                    "BeneficiaryClaimCount"
                ]
                .sum(),
            )

            result = result.join(
                beneficiary_summary,
                how="left",
            )

        # --------------------------------------------------------
        # Monthly activity
        # --------------------------------------------------------

        if "ClaimStartDt" in claims.columns:

            dated = claims[
                [
                    "Provider",
                    "ClaimStartDt",
                ]
            ].dropna()

            if not dated.empty:

                dated["ClaimMonth"] = (
                    dated["ClaimStartDt"]
                    .dt.to_period("M")
                )

                monthly = (
                    dated.groupby(
                        ["Provider", "ClaimMonth"]
                    )
                    .size()
                    .rename("MonthlyClaimCount")
                    .reset_index()
                )

                monthly_summary = (
                    monthly
                    .groupby("Provider")
                    .agg(
                        ActiveClaimMonths=(
                            "ClaimMonth",
                            "nunique",
                        ),
                        AverageMonthlyClaims=(
                            "MonthlyClaimCount",
                            "mean",
                        ),
                        MaximumMonthlyClaims=(
                            "MonthlyClaimCount",
                            "max",
                        ),
                        MonthlyClaimStd=(
                            "MonthlyClaimCount",
                            "std",
                        ),
                    )
                )

                monthly_summary[
                    "MonthlyClaimCV"
                ] = self._safe_divide(
                    monthly_summary[
                        "MonthlyClaimStd"
                    ],
                    monthly_summary[
                        "AverageMonthlyClaims"
                    ],
                )

                result = result.join(
                    monthly_summary,
                    how="left",
                )

                activity_span = (
                    dated.groupby("Provider")[
                        "ClaimStartDt"
                    ]
                    .agg(
                        FirstClaimDate="min",
                        LastClaimDate="max",
                    )
                )

                activity_span[
                    "ProviderActivityDays"
                ] = (
                    activity_span["LastClaimDate"]
                    - activity_span["FirstClaimDate"]
                ).dt.days

                result = result.join(
                    activity_span[
                        ["ProviderActivityDays"]
                    ],
                    how="left",
                )

        # Helper columns are not needed in the model.
        result = result.drop(
            columns=[
                "ClaimAmountQ25",
                "ClaimAmountQ75",
                "ClaimAmountMin",
                "ClaimAmountMax",
                "ClaimAmountMeanAdvanced",
                "ClaimAmountMedianAdvanced",
                "ClaimAmountStdAdvanced",
                "ClaimDurationMedianAdvanced",
                "ClaimDurationStdAdvanced",
            ],
            errors="ignore",
        )

        return result.reset_index()

    # ============================================================
    # ACTIVITY FEATURES
    # ============================================================

    def _create_activity_features(
        self,
        inpatient_df: pd.DataFrame,
        outpatient_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates provider inpatient/outpatient activity indicators.
        """

        providers = set()

        for dataframe in [
            inpatient_df,
            outpatient_df,
        ]:
            if not dataframe.empty:
                providers.update(
                    dataframe["Provider"]
                    .dropna()
                    .astype(str)
                    .unique()
                )

        if not providers:
            return pd.DataFrame(
                columns=[
                    "Provider",
                    "HasInpatientActivity",
                    "HasOutpatientActivity",
                ]
            )

        result = pd.DataFrame(
            {"Provider": sorted(providers)}
        )

        inpatient_counts = (
            inpatient_df.groupby("Provider").size()
            if not inpatient_df.empty
            else pd.Series(dtype=float)
        )

        outpatient_counts = (
            outpatient_df.groupby("Provider").size()
            if not outpatient_df.empty
            else pd.Series(dtype=float)
        )

        result["InpatientActivityCount"] = (
            result["Provider"]
            .map(inpatient_counts)
            .fillna(0)
        )

        result["OutpatientActivityCount"] = (
            result["Provider"]
            .map(outpatient_counts)
            .fillna(0)
        )

        result["TotalActivityClaims"] = (
            result["InpatientActivityCount"]
            + result["OutpatientActivityCount"]
        )

        result["HasInpatientActivity"] = (
            result["InpatientActivityCount"] > 0
        ).astype(int)

        result["HasOutpatientActivity"] = (
            result["OutpatientActivityCount"] > 0
        ).astype(int)

        result["InpatientActivityRatio"] = (
            self._safe_divide(
                result["InpatientActivityCount"],
                result["TotalActivityClaims"],
            )
        )

        result["OutpatientActivityRatio"] = (
            self._safe_divide(
                result["OutpatientActivityCount"],
                result["TotalActivityClaims"],
            )
        )

        return result

    # ============================================================
    # PEER-RELATIVE FEATURES
    # ============================================================

    def _create_relative_peer_features(
        self,
        provider_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates provider-level percentile and peer-median features.

        These features use only provider behavior and never use
        PotentialFraud.
        """

        result = provider_features[
            ["Provider"]
        ].copy()

        candidate_features = [
            "TotalClaims",
            "TotalReimbursement",
            "TotalDeductiblePaid",
            "TotalAverageClaim",
            "ClaimsPerBeneficiary",
            "ClaimsPerAttendingPhysician",
            "ClaimsPerOperatingPhysician",
            "MaximumHospitalStay",
            "MaximumClaimAmount",
            "HighValueClaimRatio",
            "ReimbursementPerBeneficiary",
            "DeductiblePerBeneficiary",
            "AverageClaimDuration",
            "UniqueBeneficiaries",
        ]

        for feature in candidate_features:

            if feature not in provider_features.columns:
                continue

            values = pd.to_numeric(
                provider_features[feature],
                errors="coerce",
            )

            result[
                f"{feature}Percentile"
            ] = values.rank(
                pct=True,
                method="average",
            )

            median = values.median()

            if pd.notna(median) and median != 0:

                result[
                    f"{feature}VsPeerMedian"
                ] = values / median

        return result

    # ============================================================
    # FINAL FEATURE TABLE
    # ============================================================

    def _create_final_features(
        self,
        provider_df: pd.DataFrame,
        inpatient_df: pd.DataFrame,
        outpatient_df: pd.DataFrame,
        beneficiary_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combines every provider-level feature group into the
        final modelling dataset.
        """

        logger.info(
            "Building final provider feature table."
        )

        # --------------------------------------------------------
        # Start with provider labels
        # --------------------------------------------------------

        provider_columns = ["Provider"]

        if "PotentialFraud" in provider_df.columns:
            provider_columns.append("PotentialFraud")

        provider_features = (
            provider_df[provider_columns]
            .drop_duplicates(
                subset=["Provider"]
            )
            .copy()
        )

        # --------------------------------------------------------
        # Claim features
        # --------------------------------------------------------

        claim_features = (
            self._aggregate_claim_features(
                inpatient_df,
                outpatient_df,
            )
        )

        provider_features = provider_features.merge(
            claim_features,
            on="Provider",
            how="left",
        )

        # --------------------------------------------------------
        # Total claims
        # --------------------------------------------------------

        provider_features[
            "InpatientClaims"
        ] = provider_features[
            "InpatientClaims"
        ].fillna(0)

        provider_features[
            "OutpatientClaims"
        ] = provider_features[
            "OutpatientClaims"
        ].fillna(0)

        provider_features[
            "TotalClaims"
        ] = (
            provider_features[
                "InpatientClaims"
            ]
            +
            provider_features[
                "OutpatientClaims"
            ]
        )

        # --------------------------------------------------------
        # Total reimbursement
        # --------------------------------------------------------

        provider_features[
            "TotalReimbursement"
        ] = (
            provider_features[
                "InpatientReimbursement"
            ].fillna(0)
            +
            provider_features[
                "OutpatientReimbursement"
            ].fillna(0)
        )

        # --------------------------------------------------------
        # Total deductible
        # --------------------------------------------------------

        provider_features[
            "TotalDeductiblePaid"
        ] = (
            provider_features[
                "TotalInpatientDeductible"
            ].fillna(0)
            +
            provider_features[
                "TotalOutpatientDeductible"
            ].fillna(0)
        )

        # --------------------------------------------------------
        # Claim ratios
        # --------------------------------------------------------

        provider_features[
            "InpatientClaimRatio"
        ] = self._safe_divide(
            provider_features[
                "InpatientClaims"
            ],
            provider_features[
                "TotalClaims"
            ],
        )

        provider_features[
            "OutpatientClaimRatio"
        ] = self._safe_divide(
            provider_features[
                "OutpatientClaims"
            ],
            provider_features[
                "TotalClaims"
            ],
        )

        # --------------------------------------------------------
        # Total average claim
        # --------------------------------------------------------

        provider_features[
            "TotalAverageClaim"
        ] = self._safe_divide(
            provider_features[
                "TotalReimbursement"
            ],
            provider_features[
                "TotalClaims"
            ],
        )

        # --------------------------------------------------------
        # Beneficiary features
        # --------------------------------------------------------

        beneficiary_features = (
            self._aggregate_beneficiary_features(
                beneficiary_df,
                inpatient_df,
                outpatient_df,
            )
        )

        provider_features = provider_features.merge(
            beneficiary_features,
            on="Provider",
            how="left",
        )

        # --------------------------------------------------------
        # Distribution features
        # --------------------------------------------------------

        distribution_features = (
            self._create_claim_distribution_features(
                inpatient_df,
                outpatient_df,
            )
        )

        provider_features = provider_features.merge(
            distribution_features,
            on="Provider",
            how="left",
            suffixes=(
                "",
                "_distribution",
            ),
        )

        # --------------------------------------------------------
        # Physician features
        # --------------------------------------------------------

        physician_features = (
            self._aggregate_physician_features(
                inpatient_df,
                outpatient_df,
            )
        )

        provider_features = provider_features.merge(
            physician_features,
            on="Provider",
            how="left",
        )

        # --------------------------------------------------------
        # Diagnosis / procedure features
        # --------------------------------------------------------

        code_features = (
            self._aggregate_diagnosis_procedure_features(
                inpatient_df,
                outpatient_df,
            )
        )

        provider_features = provider_features.merge(
            code_features,
            on="Provider",
            how="left",
        )

        # --------------------------------------------------------
        # Advanced behavioral features
        # --------------------------------------------------------

        advanced_features = (
            self._create_advanced_behavior_features(
                inpatient_df,
                outpatient_df,
            )
        )

        provider_features = provider_features.merge(
            advanced_features,
            on="Provider",
            how="left",
            suffixes=("", "_advanced"),
        )

        # --------------------------------------------------------
        # Activity features
        # --------------------------------------------------------

        activity_features = (
            self._create_activity_features(
                inpatient_df,
                outpatient_df,
            )
        )

        provider_features = provider_features.merge(
            activity_features,
            on="Provider",
            how="left",
        )

        # --------------------------------------------------------
        # High-value claim features
        # --------------------------------------------------------

        all_claims = []

        if not inpatient_df.empty:

            inpatient_claims = (
                inpatient_df[
                    [
                        "Provider",
                        "ClaimAmount",
                    ]
                ]
                .copy()
            )

            inpatient_claims[
                "ClaimType"
            ] = "Inpatient"

            all_claims.append(
                inpatient_claims
            )

        if not outpatient_df.empty:

            outpatient_claims = (
                outpatient_df[
                    [
                        "Provider",
                        "ClaimAmount",
                    ]
                ]
                .copy()
            )

            outpatient_claims[
                "ClaimType"
            ] = "Outpatient"

            all_claims.append(
                outpatient_claims
            )

        if all_claims:

            all_claim_data = pd.concat(
                all_claims,
                ignore_index=True,
            )

            # ----------------------------------------------------
            # Overall 95th percentile
            # ----------------------------------------------------

            overall_threshold = (
                all_claim_data[
                    "ClaimAmount"
                ].quantile(0.95)
            )

            all_claim_data[
                "HighValueClaim"
            ] = (
                all_claim_data[
                    "ClaimAmount"
                ]
                > overall_threshold
            ).astype(int)

            high_value_features = (
                all_claim_data
                .groupby("Provider")
                .agg(
                    HighValueClaimCount=(
                        "HighValueClaim",
                        "sum",
                    ),

                    HighValueClaimRatio=(
                        "HighValueClaim",
                        "mean",
                    ),
                )
            )

            provider_features = (
                provider_features.merge(
                    high_value_features.reset_index(),
                    on="Provider",
                    how="left",
                )
            )

        # --------------------------------------------------------
        # Peer-relative behavioral features
        # --------------------------------------------------------

        peer_features = (
            self._create_relative_peer_features(
                provider_features
            )
        )

        provider_features = provider_features.merge(
            peer_features,
            on="Provider",
            how="left",
            suffixes=("", "_peer"),
        )

        # --------------------------------------------------------
        # Remove duplicate/helper columns
        # --------------------------------------------------------

        helper_columns = [
            "TotalInpatientDeductible",
            "TotalOutpatientDeductible",
            "AverageBeneficiaryClaims",
        ]

        provider_features = (
            provider_features.drop(
                columns=[
                    column
                    for column in helper_columns
                    if column in provider_features.columns
                ],
                errors="ignore",
            )
        )

        # --------------------------------------------------------
        # Remove accidental duplicate columns
        # --------------------------------------------------------

        provider_features = (
            provider_features.loc[
                :,
                ~provider_features.columns.duplicated(),
            ]
        )

        # --------------------------------------------------------
        # Replace infinite values
        # --------------------------------------------------------

        provider_features = (
            provider_features.replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        # --------------------------------------------------------
        # Remove redundant features
        # --------------------------------------------------------

        redundant_features = [
            "OutpatientReimbursementRatio",
            "DurationVariationRatio",
        ]

        provider_features = provider_features.drop(
            columns=[
                column
                for column in redundant_features
                if column in provider_features.columns
            ],
            errors="ignore",
        )

        # --------------------------------------------------------
        # Sort by provider
        # --------------------------------------------------------

        provider_features = (
            provider_features
            .sort_values(
                "Provider"
            )
            .reset_index(
                drop=True
            )
        )

        logger.info(
            "Final provider feature shape: %s",
            provider_features.shape,
        )

        logger.info(
            "Final feature count: %d",
            len(
                provider_features.columns
            ),
        )

        # --------------------------------------------------------
        # Final quality checks
        # --------------------------------------------------------

        if provider_features.columns.duplicated().any():
            raise ValueError(
                "Duplicate columns found in final feature dataset."
            )

        if provider_features[
            "Provider"
        ].duplicated().any():
            raise ValueError(
                "Duplicate providers found in final feature dataset."
            )

        logger.info(
            "Duplicate rows: %d",
            provider_features.duplicated().sum(),
        )

        logger.info(
            "Missing values: %d",
            provider_features.isna().sum().sum(),
        )

        return provider_features

    # ============================================================
    # PUBLIC METHOD
    # ============================================================

    def build_provider_features(
        self,
        provider_df: pd.DataFrame,
        beneficiary_df: pd.DataFrame,
        inpatient_df: pd.DataFrame,
        outpatient_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Main feature-engineering entry point.

        Calling this one method executes the complete feature
        engineering pipeline.
        """

        try:

            logger.info(
                "Starting provider feature engineering."
            )

            # ----------------------------------------------------
            # Validate provider dataset
            # ----------------------------------------------------

            # PotentialFraud is required for training data, but it is
            # not present in unseen test/prediction data.
            if "Provider" not in provider_df.columns:

                raise ValueError(
                    "Provider dataset must contain Provider."
                )

            has_target = "PotentialFraud" in provider_df.columns

            logger.info(
                "PotentialFraud present: %s",
                has_target,
            )

            # ----------------------------------------------------
            # Validate beneficiary dataset
            # ----------------------------------------------------

            if "BeneID" not in beneficiary_df.columns:

                raise ValueError(
                    "Beneficiary dataset must contain BeneID."
                )

            # ----------------------------------------------------
            # Validate inpatient dataset
            # ----------------------------------------------------

            required_inpatient_columns = {
                "Provider",
                "BeneID",
                "ClaimID",
            }

            missing_inpatient = (
                required_inpatient_columns
                - set(
                    inpatient_df.columns
                )
            )

            if missing_inpatient:

                raise ValueError(
                    "Inpatient dataset is missing "
                    f"columns: {missing_inpatient}"
                )

            # ----------------------------------------------------
            # Validate outpatient dataset
            # ----------------------------------------------------

            required_outpatient_columns = {
                "Provider",
                "BeneID",
                "ClaimID",
            }

            missing_outpatient = (
                required_outpatient_columns
                - set(
                    outpatient_df.columns
                )
            )

            if missing_outpatient:

                raise ValueError(
                    "Outpatient dataset is missing "
                    f"columns: {missing_outpatient}"
                )

            # ----------------------------------------------------
            # Copy inputs
            # ----------------------------------------------------

            provider_df = provider_df.copy()
            beneficiary_df = beneficiary_df.copy()
            inpatient_df = inpatient_df.copy()
            outpatient_df = outpatient_df.copy()

            # ----------------------------------------------------
            # Remove accidental duplicate rows
            # ----------------------------------------------------

            provider_df = provider_df.drop_duplicates()

            beneficiary_df = (
                beneficiary_df.drop_duplicates(
                    subset=["BeneID"]
                )
            )

            inpatient_df = (
                inpatient_df.drop_duplicates(
                    subset=["ClaimID"]
                )
            )

            outpatient_df = (
                outpatient_df.drop_duplicates(
                    subset=["ClaimID"]
                )
            )

            # ----------------------------------------------------
            # Prepare dates
            # ----------------------------------------------------

            (
                beneficiary_df,
                inpatient_df,
                outpatient_df,
            ) = self._prepare_dates(
                beneficiary_df,
                inpatient_df,
                outpatient_df,
            )

            # ----------------------------------------------------
            # Claim-level features
            # ----------------------------------------------------

            inpatient_df = (
                self._create_inpatient_features(
                    inpatient_df
                )
            )

            outpatient_df = (
                self._create_outpatient_features(
                    outpatient_df
                )
            )

            # ----------------------------------------------------
            # Beneficiary-level features
            # ----------------------------------------------------

            beneficiary_df = (
                self._create_beneficiary_features(
                    beneficiary_df,
                    inpatient_df,
                    outpatient_df,
                )
            )

            # ----------------------------------------------------
            # Provider-level features
            # ----------------------------------------------------

            provider_features = (
                self._create_final_features(
                    provider_df,
                    inpatient_df,
                    outpatient_df,
                    beneficiary_df,
                )
            )

            logger.info(
                "Provider feature engineering completed successfully."
            )

            return provider_features

        except Exception as exc:

            logger.exception(
                "Feature engineering failed."
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
        "FeatureEngineering module loaded successfully."
    )