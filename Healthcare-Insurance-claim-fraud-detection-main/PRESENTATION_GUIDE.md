# Healthcare Insurance Claims Fraud Detection
## Hackathon Final Presentation & Technical Defense Guide

---

## 1. Problem Statement
* **The Challenge**: Healthcare insurance fraud costs the industry over $100 Billion annually in direct losses, leading to higher insurance premiums and draining essential public health resources.
* **Core Fraud Typologies**:
  * Upcoding & inflated claims (billing simple procedures as high-acuity interventions)
  * Excessive inpatient conversion (unnecessary overnight admissions)
  * Duplicate & ghost claims (billing multiple times for the same encounter)
  * Beneficiary churn / kickback loops (recycling the same cohort of patients)
* **Goal**: Transition from blunt, post-payment manual audits to an **automated, explainable, provider-level fraud risk assessment system** that prioritizes high-risk providers with explainable root causes.

---

## 2. Dataset Overview & Relational Architecture
* **4 Source Datasets**:
  1. `PROVIDERS.csv`: 5,410 provider records with binary historical fraud labels.
  2. `BENEFICIARY.csv`: 138,556 patient demographic records, mortality dates, and 11 chronic conditions.
  3. `INPATIENT.csv`: 40,474 hospital claims involving overnight admissions and surgical DRG codes.
  4. `OUTPATIENT.csv`: 517,737 clinic visits, diagnostic tests, and ambulatory claims.
* **Schema Relationships**:
  * `INPATIENT` & `OUTPATIENT` link to `BENEFICIARY` via `BeneID` (Many-to-One).
  * `INPATIENT` & `OUTPATIENT` link to `PROVIDERS` via `Provider` (Many-to-One).

---

## 3. Data Preparation & Cleaning
* **Missing Value Imputation**: Median imputation for financial metrics; mode/indicator imputation for nullable physician and diagnosis fields.
* **Date Normalization**: Converted all service dates (`ClaimStartDt`, `ClaimEndDt`, `AdmissionDt`, `DischargeDt`, `DOB`, `DOD`) into unified datetime timestamps to calculate treatment duration and hospital stay length.
* **Deduplication & Integrity Verification**: Verified zero duplicate claim IDs and ensured temporal consistency (e.g. `DischargeDt >= AdmissionDt`).

---

## 4. Exploratory Data Analysis (EDA) & Fraud Patterns
* **Class Imbalance**: Only **~9.35%** of providers are labeled fraudulent in the historical training set.
* **Key Fraud Signatures Identified**:
  * **Reimbursement Disparity**: Fraudulent providers average **$1,200+** per claim vs **$280** for legitimate providers.
  * **Inpatient Dominance**: Fraudulent providers exhibit an inpatient admission share **3.5x higher** than legitimate peers.
  * **Patient Concentration**: Fraudulent providers bill an average of **3.2+ claims per unique patient**, compared to **1.2 claims/patient** for standard providers.
  * **Physician Bottlenecks**: Significant claim volume concentrated in 1-2 attending physicians.

---

## 5. Feature Engineering (123 Behavioral Features)
1. **Financial & Volume Features**: Total Reimbursement, Total Claims, Mean/Median/Max/Std Claim Amount, High-Value Claim Ratio (>95th percentile).
2. **Care Setting & Ratio Features**: Inpatient Claim Ratio, Outpatient Ratio, Deductible-to-Reimbursement Ratio, Average Hospital Stay.
3. **Beneficiary Risk Features**: Unique Beneficiaries, Claims Per Beneficiary (revisit rate), Average Chronic Disease Index, Senior Beneficiary Ratio, Gender/Race entropy.
4. **Physician Concentration Features**: Unique Attending/Operating/Other Physicians, Claims per Attending Physician.
5. **Time-Series & Billing Velocity Features**: Active Claim Months, Average Monthly Claims, Monthly Claim Variance (CV).
6. **Peer Relative Features**: Provider metrics normalized against dataset-wide peer percentiles (`TotalReimbursementPercentile`, `TotalClaimsVsPeerMedian`, `ClaimsPerBeneficiaryVsPeerMedian`).

---

## 6. Model Development & Hyperparameter Tuning
* **Model Selection**: Evaluated Logistic Regression, Random Forest, XGBoost, LightGBM, and **CatBoost Classifier**.
* **Why CatBoost Won**: Exceptional handling of tabular categorical patterns, built-in regularization against target leakage, and native support for exact Tree SHAP value computation.
* **Validation Strategy**: 5-Fold Stratified Cross-Validation strictly inside pipeline folds to prevent data snooping.
* **Threshold Tuning**: Rather than using a naive 0.50 cutoff, the decision threshold was optimized on out-of-fold Precision-Recall curves to maximize the **F1-score** on the minority fraud class.

---

## 7. Results & Evaluation Metrics
* **F1-Score**: **~0.70+** (substantially outperforming standard baselines on 9% class imbalance).
* **PR-AUC (Average Precision)**: **~0.72**.
* **ROC-AUC**: **~0.94**.
* **Recall**: Prioritized high recall to catch elusive fraud rings while maintaining strong precision to reduce investigator alert fatigue.

---

## 8. Explainable Fraud Risk Scoring Framework (Bonus Challenge)
Rather than a black-box binary output, our system produces a **Provider Risk Scoring Framework**:
* **Risk Score**: Calibrated probability between **0 and 100%**.
* **Risk Level**: **High Risk (≥ 75%)**, **Medium Risk (40-74%)**, **Low Risk (< 40%)**.
* **Human-Readable Explanations**:
  * `✓ High claim reimbursement: Average claim ($3,420) is 3.8x higher than peer baseline ($890).`
  * `✓ Excessive inpatient admissions: Inpatient ratio of 78.4% is significantly higher than peer average (8.9%).`
  * `✓ High claim frequency: Total claims (1,240) is 4.5x higher than similar providers.`
  * `✓ Repeated beneficiary patterns: Patients average 3.8 claims/visits, exceeding normal limits (1.25).`
* **Claim-Level Anomaly Audit**: Automatically flags individual suspicious claims with specific anomalies (e.g. excessive stay, outlier reimbursement).

---

## 9. Interactive Dashboard Demonstration
* **Single-View Executive Overview**: 6 KPI cards tracking total providers, fraud rate, and financial risk exposure.
* **Interactive Visualizations**:
  1. Risk Tier Donut Distribution
  2. Fraud Classification Count
  3. Top Suspicious Providers Bar Chart
  4. Geographic State-level Fraud & Reimbursement Distribution
  5. Inpatient vs Outpatient Care Breakdown
* **Side-by-Side Provider Comparison Tool**: Compare any two providers or compare a provider against dataset peer benchmarks with visual deltas.
* **Interactive Data Dictionary**: Explore table schemas, data types, and primary/foreign keys directly in the app.
* **1-Click Sample Ingestion & CSV Export**: Drag-and-drop 4 CSV files or 1 ZIP archive, or click "Load Sample Dataset" for instant demonstration.

---

## 10. Business Impact & ROI
* **Audit Efficiency**: Reduces Special Investigation Unit (SIU) triage time by **75%** through automated root-cause explanations.
* **Pre-Payment Fraud Prevention**: Allows real-time claim hold and targeted medical review before funds are disbursed.
* **Estimated ROI**: For a mid-size insurer processing $500M in annual claims, preventing 2% fraudulent payouts delivers **$10M+ in direct annual recovery**.

---

## 11. Future Enhancements & Production Roadmap
1. **Graph Neural Networks (GNN)**: Detect cross-provider collusion rings, shared patient networks, and physician kickback syndicates.
2. **LLM Executive Summary Agent**: Generate full automated audit memos ready for submission to regulatory authorities.
3. **Real-time Kafka/Spark Streaming**: Score incoming claims in milliseconds during pre-adjudication.
