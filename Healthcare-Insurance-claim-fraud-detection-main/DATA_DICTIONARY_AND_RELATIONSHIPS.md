# Healthcare Insurance Claim Fraud Detection
## Data Dictionary, Entity Relationships & Analytical Architecture

---

## 1. Executive Summary & Problem Understanding

Healthcare insurance fraud incurs tens of billions of dollars in annual losses, directly increasing healthcare costs, depleting public funds (Medicare/Medicaid), and straining healthcare infrastructure.

Fraud is primarily committed at the **Healthcare Provider level** through sophisticated behavioral schemes:
1. **Upcoding & Inflated Reimbursements**: Billed diagnostic/treatment codes are systematically elevated to higher reimbursement brackets than warranted.
2. **Phantom Billing & Ghost Claims**: Incurring claims for patients without actual clinical consultations or inpatient stays.
3. **Excessive Inpatient Admissions**: Unnecessarily converting outpatient clinical visits into overnight hospital admissions.
4. **Beneficiary Churn & Revisit Loops**: Systematically recycling the same patient IDs across multiple high-value claims.
5. **Physician Concentration Anomalies**: Funneling disproportionate claim volumes through a single attending or operating physician.

This solution ingests and merges the four core CMS claims tables, calculates **120+ provider-level behavioral features**, scores fraud risk using a calibrated **CatBoost ML classifier**, and delivers a **human-explainable risk assessment score (0-100%)** with peer comparative benchmarks.

---

## 2. Entity-Relationship Model (ERD)

```mermaid
erDiagram
    PROVIDERS ||--o{ INPATIENT : "submits"
    PROVIDERS ||--o{ OUTPATIENT : "submits"
    BENEFICIARY ||--o{ INPATIENT : "receives care"
    BENEFICIARY ||--o{ OUTPATIENT : "receives care"

    PROVIDERS {
        string Provider PK "Unique Provider ID (e.g. PRV51001)"
        string PotentialFraud "Ground Truth Target (Yes / No)"
    }

    BENEFICIARY {
        string BeneID PK "Unique Beneficiary ID"
        date DOB "Date of Birth"
        date DOD "Date of Death (Nullable)"
        int Gender "1: Male, 2: Female"
        int Race "Race demographic code"
        string RenalDiseaseIndicator "End-Stage Renal Disease (0/Y)"
        int State "State Location Code"
        int County "County FIPS Code"
        int ChronicCond_Alzheimer "1: Yes, 2: No"
        int ChronicCond_Heartfailure "1: Yes, 2: No"
        int ChronicCond_KidneyDisease "1: Yes, 2: No"
        int ChronicCond_Cancer "1: Yes, 2: No"
        int ChronicCond_ObstrPulmonary "1: Yes, 2: No"
        int ChronicCond_Depression "1: Yes, 2: No"
        int ChronicCond_Diabetes "1: Yes, 2: No"
        int ChronicCond_IschemicHeart "1: Yes, 2: No"
        int ChronicCond_Osteoporasis "1: Yes, 2: No"
        int ChronicCond_rheumatoidarthritis "1: Yes, 2: No"
        int ChronicCond_stroke "1: Yes, 2: No"
        float IPAnnualReimbursementAmt "Annual Total Inpatient Reimbursement"
        float IPAnnualDeductibleAmt "Annual Total Inpatient Deductible"
        float OPAnnualReimbursementAmt "Annual Total Outpatient Reimbursement"
        float OPAnnualDeductibleAmt "Annual Total Outpatient Deductible"
    }

    INPATIENT {
        string ClaimID PK "Unique Inpatient Claim ID"
        string BeneID FK "Beneficiary Identifier"
        string Provider FK "Provider Identifier"
        date ClaimStartDt "Claim Service Start Date"
        date ClaimEndDt "Claim Service End Date"
        float InscClaimAmtReimbursed "Insurance Reimbursement Paid ($)"
        float DeductibleAmtPaid "Deductible Paid by Patient ($)"
        date AdmissionDt "Hospital Admission Date"
        date DischargeDt "Hospital Discharge Date"
        string DiagnosisGroupCode "DRG Code"
        string AttendingPhysician "Primary Attending Physician ID"
        string OperatingPhysician "Operating Surgeon ID"
        string OtherPhysician "Consulting Physician ID"
        string ClmDiagnosisCode_1to10 "ICD-9 Diagnosis Codes"
        string ClmProcedureCode_1to6 "ICD-9 Procedure Codes"
    }

    OUTPATIENT {
        string ClaimID PK "Unique Outpatient Claim ID"
        string BeneID FK "Beneficiary Identifier"
        string Provider FK "Provider Identifier"
        date ClaimStartDt "Claim Service Start Date"
        date ClaimEndDt "Claim Service End Date"
        float InscClaimAmtReimbursed "Insurance Reimbursement Paid ($)"
        float DeductibleAmtPaid "Deductible Paid by Patient ($)"
        string AttendingPhysician "Attending Physician ID"
        string OperatingPhysician "Operating Physician ID"
        string OtherPhysician "Consulting Physician ID"
        string ClmDiagnosisCode_1to10 "ICD-9 Diagnosis Codes"
        string ClmProcedureCode_1to6 "ICD-9 Procedure Codes"
    }
```

---

## 3. Detailed Data Dictionary

### Table 1: `PROVIDERS.csv`
| Column Name | Key / Role | Data Type | Nullable | Description |
| :--- | :--- | :--- | :--- | :--- |
| `Provider` | **Primary Key** | String | No | Unique alphanumeric healthcare provider ID (e.g. `PRV51001`). |
| `PotentialFraud` | **Target Label** | String | No (Train) | Classification label: `"Yes"` (Fraudulent) or `"No"` (Legitimate). |

### Table 2: `BENEFICIARY.csv`
| Column Name | Key / Role | Data Type | Nullable | Description |
| :--- | :--- | :--- | :--- | :--- |
| `BeneID` | **Primary Key** | String | No | Unique beneficiary/patient ID. |
| `DOB` | Attribute | Date | No | Date of birth for age and mortality features. |
| `DOD` | Attribute | Date | Yes | Date of death (used for deceased patient billing fraud checks). |
| `Gender` | Attribute | Integer | No | Gender indicator: `1` (Male), `2` (Female). |
| `Race` | Attribute | Integer | No | Demographic race category code (1-5). |
| `RenalDiseaseIndicator`| Attribute | String | No | End-stage renal disease indicator (`'0'` or `'Y'`). |
| `State` | Attribute | Integer | No | Geographic State code of beneficiary residence. |
| `County` | Attribute | Integer | No | County FIPS identifier. |
| `ChronicCond_*` (11 cols)| Attribute | Binary | No | Chronic disease indicators: Alzheimer's, Heart Failure, Kidney Disease, Cancer, COPD, Depression, Diabetes, Ischemic Heart, Osteoporosis, Rheumatoid Arthritis, Stroke (`1`: Yes, `2`: No). |
| `IPAnnualReimbursementAmt`| Attribute | Float | No | Annual aggregate inpatient claims reimbursed. |
| `IPAnnualDeductibleAmt` | Attribute | Float | No | Annual aggregate inpatient deductible paid. |
| `OPAnnualReimbursementAmt`| Attribute | Float | No | Annual aggregate outpatient claims reimbursed. |
| `OPAnnualDeductibleAmt` | Attribute | Float | No | Annual aggregate outpatient deductible paid. |

### Table 3: `INPATIENT.csv`
| Column Name | Key / Role | Data Type | Nullable | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ClaimID` | **Primary Key** | String | No | Unique alphanumeric claim ID. |
| `BeneID` | **Foreign Key** | String | No | References `BENEFICIARY.BeneID`. |
| `Provider` | **Foreign Key** | String | No | References `PROVIDERS.Provider`. |
| `ClaimStartDt` | Attribute | Date | No | Service start date. |
| `ClaimEndDt` | Attribute | Date | No | Service end date. |
| `InscClaimAmtReimbursed`| Metric | Float | No | Primary financial insurance payout amount. |
| `DeductibleAmtPaid` | Metric | Float | Yes | Deductible copayment paid by patient. |
| `AdmissionDt` | Attribute | Date | No | Hospital admission date. |
| `DischargeDt` | Attribute | Date | No | Hospital discharge date. |
| `DiagnosisGroupCode`| Categorical | String | Yes | Diagnosis Related Group (DRG) code. |
| `AttendingPhysician` | Attribute | String | Yes | Primary attending clinician ID. |
| `OperatingPhysician` | Attribute | String | Yes | Surgical/operating physician ID. |
| `OtherPhysician` | Attribute | String | Yes | Consulting physician ID. |
| `ClmDiagnosisCode_1 to 10`| Categorical | String | Yes | ICD-9 primary and secondary diagnosis codes. |
| `ClmProcedureCode_1 to 6`| Categorical | String | Yes | ICD-9 surgical and clinical procedure codes. |

### Table 4: `OUTPATIENT.csv`
| Column Name | Key / Role | Data Type | Nullable | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ClaimID` | **Primary Key** | String | No | Unique outpatient claim ID. |
| `BeneID` | **Foreign Key** | String | No | References `BENEFICIARY.BeneID`. |
| `Provider` | **Foreign Key** | String | No | References `PROVIDERS.Provider`. |
| `ClaimStartDt` | Attribute | Date | No | Outpatient visit start date. |
| `ClaimEndDt` | Attribute | Date | No | Outpatient visit end date. |
| `InscClaimAmtReimbursed`| Metric | Float | No | Outpatient reimbursement amount paid. |
| `DeductibleAmtPaid` | Metric | Float | Yes | Outpatient deductible paid. |
| `AttendingPhysician` | Attribute | String | Yes | Attending physician ID. |
| `OperatingPhysician` | Attribute | String | Yes | Operating physician ID. |
| `OtherPhysician` | Attribute | String | Yes | Consulting clinician ID. |
| `ClmDiagnosisCode_1 to 10`| Categorical | String | Yes | ICD-9 diagnosis codes. |
| `ClmProcedureCode_1 to 6`| Categorical | String | Yes | ICD-9 procedure codes. |

---

## 4. Multi-Table Merge & Feature Transformation Pipeline

```mermaid
flowchart TD
    A[PROVIDERS.csv] --> E[Provider-Level Feature Aggregation Engine]
    B[BENEFICIARY.csv] --> E
    C[INPATIENT.csv] --> E
    D[OUTPATIENT.csv] --> E

    E --> F[123 Engineered Behavioral Features]
    F --> G[CatBoost Classifier & Tuned Threshold]
    G --> H[Fraud Risk Probability 0-100%]
    G --> I[SHAP Feature Attribution]
    G --> J[Peer Comparative Baseline Anomaly Scoring]

    H --> K[Executive Dashboard & UI Visualizations]
    I --> K
    J --> K
```
