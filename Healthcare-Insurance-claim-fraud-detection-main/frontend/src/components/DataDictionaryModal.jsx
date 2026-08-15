import React, { useState } from 'react';
import { 
  X, 
  Database, 
  Key, 
  Link2, 
  FileText, 
  Layers, 
  Table, 
  CheckCircle2, 
  Sparkles,
  ArrowRight,
  ShieldAlert
} from 'lucide-react';

export default function DataDictionaryModal({ onClose }) {
  const [selectedTable, setSelectedTable] = useState('PROVIDERS');

  const tables = {
    PROVIDERS: {
      name: 'Providers Dataset',
      file: 'PROVIDERS.csv',
      description: 'Contains healthcare provider identifiers and supervised fraud ground-truth labels.',
      badge: 'Provider Level',
      primaryKey: 'Provider',
      foreignKeys: [],
      columns: [
        { name: 'Provider', type: 'String', key: 'PK', desc: 'Unique alphanumeric identifier assigned to healthcare provider (e.g. PRV51001).' },
        { name: 'PotentialFraud', type: 'Categorical', key: 'Target', desc: 'Supervised classification label: "Yes" (Fraudulent) or "No" (Legitimate).' }
      ]
    },
    BENEFICIARY: {
      name: 'Beneficiary Dataset',
      file: 'BENEFICIARY.csv',
      description: 'Comprehensive patient demographics, geographic location, coverage periods, and 11 chronic condition indicators.',
      badge: 'Patient Level',
      primaryKey: 'BeneID',
      foreignKeys: [],
      columns: [
        { name: 'BeneID', type: 'String', key: 'PK', desc: 'Unique alphanumeric identifier for the insured patient/beneficiary.' },
        { name: 'DOB', type: 'Date', key: '', desc: 'Date of Birth (used to calculate age at time of claim).' },
        { name: 'DOD', type: 'Date (Nullable)', key: '', desc: 'Date of Death (used to identify claims billed after deceased date).' },
        { name: 'Gender', type: 'Integer (1/2)', key: '', desc: '1: Male, 2: Female.' },
        { name: 'Race', type: 'Integer (1-5)', key: '', desc: 'Demographic race code.' },
        { name: 'RenalDiseaseIndicator', type: 'String (0/Y)', key: '', desc: 'Indicator of End-Stage Renal Disease (ESRD).' },
        { name: 'State', type: 'Integer', key: '', desc: 'Geographic state code where the beneficiary resides.' },
        { name: 'County', type: 'Integer', key: '', desc: 'County FIPS code.' },
        { name: 'ChronicCond_Alzheimer', type: 'Binary (1/2)', key: '', desc: '1: Diagnosed with Alzheimer’s Disease, 2: No.' },
        { name: 'ChronicCond_Heartfailure', type: 'Binary (1/2)', key: '', desc: '1: Diagnosed with Chronic Heart Failure, 2: No.' },
        { name: 'ChronicCond_KidneyDisease', type: 'Binary (1/2)', key: '', desc: '1: Chronic Kidney Disease diagnosed, 2: No.' },
        { name: 'ChronicCond_Cancer', type: 'Binary (1/2)', key: '', desc: '1: Oncological Cancer diagnosis, 2: No.' },
        { name: 'ChronicCond_ObstrPulmonary', type: 'Binary (1/2)', key: '', desc: '1: Chronic Obstructive Pulmonary Disease (COPD), 2: No.' },
        { name: 'ChronicCond_Depression', type: 'Binary (1/2)', key: '', desc: '1: Major Depressive Disorder diagnosed, 2: No.' },
        { name: 'ChronicCond_Diabetes', type: 'Binary (1/2)', key: '', desc: '1: Diagnosed with Type 1 or Type 2 Diabetes, 2: No.' },
        { name: 'ChronicCond_IschemicHeart', type: 'Binary (1/2)', key: '', desc: '1: Ischemic / Coronary Artery Disease, 2: No.' },
        { name: 'ChronicCond_Osteoporasis', type: 'Binary (1/2)', key: '', desc: '1: Osteoporosis diagnosis, 2: No.' },
        { name: 'ChronicCond_rheumatoidarthritis', type: 'Binary (1/2)', key: '', desc: '1: Rheumatoid Arthritis diagnosed, 2: No.' },
        { name: 'ChronicCond_stroke', type: 'Binary (1/2)', key: '', desc: '1: History of Stroke / Cerebrovascular accident, 2: No.' },
        { name: 'IPAnnualReimbursementAmt', type: 'Float ($)', key: '', desc: 'Total historical inpatient reimbursement paid across the year.' },
        { name: 'IPAnnualDeductibleAmt', type: 'Float ($)', key: '', desc: 'Total historical inpatient deductible paid.' },
        { name: 'OPAnnualReimbursementAmt', type: 'Float ($)', key: '', desc: 'Total historical outpatient reimbursement paid.' },
        { name: 'OPAnnualDeductibleAmt', type: 'Float ($)', key: '', desc: 'Total historical outpatient deductible paid.' }
      ]
    },
    INPATIENT: {
      name: 'Inpatient Claims Dataset',
      file: 'INPATIENT.csv',
      description: 'Hospital admissions with overnight stays, admission/discharge dates, diagnosis codes, and attending/operating physician IDs.',
      badge: 'Hospital Claims',
      primaryKey: 'ClaimID',
      foreignKeys: ['BeneID -> BENEFICIARY.BeneID', 'Provider -> PROVIDERS.Provider'],
      columns: [
        { name: 'ClaimID', type: 'String', key: 'PK', desc: 'Unique primary key for the inpatient claim.' },
        { name: 'BeneID', type: 'String', key: 'FK', desc: 'Foreign Key referencing BENEFICIARY table.' },
        { name: 'Provider', type: 'String', key: 'FK', desc: 'Foreign Key referencing PROVIDERS table.' },
        { name: 'ClaimStartDt', type: 'Date', key: '', desc: 'Service billing start date.' },
        { name: 'ClaimEndDt', type: 'Date', key: '', desc: 'Service billing end date.' },
        { name: 'InscClaimAmtReimbursed', type: 'Float ($)', key: '', desc: 'Total reimbursement amount paid to the provider.' },
        { name: 'DeductibleAmtPaid', type: 'Float ($)', key: '', desc: 'Deductible amount paid by the beneficiary.' },
        { name: 'AdmissionDt', type: 'Date', key: '', desc: 'Formal hospital admission date.' },
        { name: 'DischargeDt', type: 'Date', key: '', desc: 'Formal hospital discharge date.' },
        { name: 'DiagnosisGroupCode', type: 'String', key: '', desc: 'Diagnosis Related Group (DRG) hospital classification code.' },
        { name: 'AttendingPhysician', type: 'String', key: '', desc: 'Physician ID responsible for primary patient oversight.' },
        { name: 'OperatingPhysician', type: 'String', key: '', desc: 'Surgeon / operating physician ID.' },
        { name: 'OtherPhysician', type: 'String', key: '', desc: 'Consulting or secondary physician ID.' },
        { name: 'ClmDiagnosisCode_1 to 10', type: 'String', key: '', desc: 'ICD-9 Clinical Diagnosis codes (primary and secondary comorbidities).' },
        { name: 'ClmProcedureCode_1 to 6', type: 'String', key: '', desc: 'ICD-9 Surgical / Treatment procedure codes.' }
      ]
    },
    OUTPATIENT: {
      name: 'Outpatient Claims Dataset',
      file: 'OUTPATIENT.csv',
      description: 'Clinical consultations, outpatient surgeries, diagnostic lab tests, and clinic visits without hospital admission.',
      badge: 'Ambulatory Claims',
      primaryKey: 'ClaimID',
      foreignKeys: ['BeneID -> BENEFICIARY.BeneID', 'Provider -> PROVIDERS.Provider'],
      columns: [
        { name: 'ClaimID', type: 'String', key: 'PK', desc: 'Unique primary key for outpatient claim.' },
        { name: 'BeneID', type: 'String', key: 'FK', desc: 'Foreign Key referencing BENEFICIARY table.' },
        { name: 'Provider', type: 'String', key: 'FK', desc: 'Foreign Key referencing PROVIDERS table.' },
        { name: 'ClaimStartDt', type: 'Date', key: '', desc: 'Service billing start date.' },
        { name: 'ClaimEndDt', type: 'Date', key: '', desc: 'Service billing end date.' },
        { name: 'InscClaimAmtReimbursed', type: 'Float ($)', key: '', desc: 'Reimbursement amount paid.' },
        { name: 'DeductibleAmtPaid', type: 'Float ($)', key: '', desc: 'Deductible amount paid by patient.' },
        { name: 'AttendingPhysician', type: 'String', key: '', desc: 'Primary attending clinician.' },
        { name: 'OperatingPhysician', type: 'String', key: '', desc: 'Specialist / operating physician.' },
        { name: 'OtherPhysician', type: 'String', key: '', desc: 'Secondary physician.' },
        { name: 'ClmDiagnosisCode_1 to 10', type: 'String', key: '', desc: 'ICD-9 Diagnosis codes.' },
        { name: 'ClmProcedureCode_1 to 6', type: 'String', key: '', desc: 'ICD-9 Procedure codes.' }
      ]
    }
  };

  const activeTable = tables[selectedTable];

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 110,
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      backdropFilter: 'blur(10px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '24px',
      overflowY: 'auto'
    }}>
      <div 
        className="glass-card animate-fade-in" 
        style={{
          width: '100%',
          maxWidth: '1100px',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          border: '1px solid var(--border-highlight)',
          boxShadow: '0 30px 70px -15px rgba(0, 0, 0, 0.8)'
        }}
      >
        {/* Modal Header */}
        <div style={{
          padding: '20px 28px',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'rgba(255, 255, 255, 0.02)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '10px',
              background: 'rgba(99, 102, 241, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Database size={22} color="var(--accent-primary)" />
            </div>
            <div>
              <h2 style={{ fontSize: '20px', fontWeight: '800' }}>
                Healthcare Claims <span className="gradient-text">Data Dictionary & Relationships</span>
              </h2>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: '2px 0 0' }}>
                Relational schema, Primary / Foreign key mappings, and field specifications
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="btn-secondary"
            style={{ padding: '8px', borderRadius: '8px' }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Entity Relationship Diagram Banner */}
        <div style={{
          padding: '16px 28px',
          background: 'rgba(99, 102, 241, 0.05)',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '13px' }}>
            <Link2 size={16} color="var(--accent-primary)" />
            <span style={{ fontWeight: '700', color: 'var(--text-primary)' }}>Schema Relationships:</span>
            <span className="badge badge-neutral" style={{ fontSize: '11px' }}>
              INPATIENT.Provider (FK) → PROVIDERS.Provider (PK)
            </span>
            <span className="badge badge-neutral" style={{ fontSize: '11px' }}>
              OUTPATIENT.Provider (FK) → PROVIDERS.Provider (PK)
            </span>
            <span className="badge badge-neutral" style={{ fontSize: '11px' }}>
              CLAIMS.BeneID (FK) → BENEFICIARY.BeneID (PK)
            </span>
          </div>
        </div>

        {/* Modal Body */}
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          
          {/* Left Table Nav */}
          <div style={{
            width: '240px',
            borderRight: '1px solid var(--border-color)',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            background: 'rgba(0, 0, 0, 0.15)'
          }}>
            <div style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.05em', marginBottom: '4px' }}>
              Select Table
            </div>
            {Object.keys(tables).map((key) => {
              const item = tables[key];
              const isSelected = selectedTable === key;
              return (
                <button
                  key={key}
                  onClick={() => setSelectedTable(key)}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-start',
                    padding: '12px 14px',
                    borderRadius: '8px',
                    border: isSelected ? '1px solid var(--accent-primary)' : '1px solid transparent',
                    background: isSelected ? 'rgba(99, 102, 241, 0.12)' : 'transparent',
                    color: isSelected ? 'white' : 'var(--text-secondary)',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    textAlign: 'left',
                    width: '100%'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                    <span style={{ fontWeight: '700', fontSize: '13px', fontFamily: 'var(--font-mono)' }}>{item.file}</span>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{item.badge}</span>
                  </div>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '3px' }}>
                    {item.columns.length} columns
                  </span>
                </button>
              );
            })}
          </div>

          {/* Right Table Details */}
          <div style={{ flex: 1, padding: '24px 28px', overflowY: 'auto' }}>
            
            {/* Table Header Info */}
            <div style={{ marginBottom: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <h3 style={{ fontSize: '18px', fontWeight: '800', margin: 0 }}>
                  {activeTable.name} <span style={{ fontFamily: 'var(--font-mono)', fontSize: '14px', color: 'var(--text-muted)' }}>({activeTable.file})</span>
                </h3>
                <span className="badge badge-primary">{activeTable.badge}</span>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', margin: '6px 0 12px' }}>
                {activeTable.description}
              </p>

              {/* Keys Info */}
              <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                <div style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px', color: '#10b981' }}>
                  <Key size={13} />
                  <span><strong>Primary Key:</strong> {activeTable.primaryKey}</span>
                </div>
                {activeTable.foreignKeys.map((fk, idx) => (
                  <div key={idx} style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px', color: '#6366f1' }}>
                    <Link2 size={13} />
                    <span><strong>Foreign Key:</strong> {fk}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Column Specification Table */}
            <div className="table-container" style={{ maxHeight: '420px', overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', background: 'rgba(255, 255, 255, 0.02)' }}>
                    <th style={{ padding: '10px 14px', color: 'var(--text-muted)', fontWeight: '600' }}>Column Name</th>
                    <th style={{ padding: '10px 14px', color: 'var(--text-muted)', fontWeight: '600' }}>Key / Role</th>
                    <th style={{ padding: '10px 14px', color: 'var(--text-muted)', fontWeight: '600' }}>Data Type</th>
                    <th style={{ padding: '10px 14px', color: 'var(--text-muted)', fontWeight: '600' }}>Description & Analytics Use</th>
                  </tr>
                </thead>
                <tbody>
                  {activeTable.columns.map((col, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '10px 14px', fontFamily: 'var(--font-mono)', fontWeight: '600', color: 'var(--text-primary)' }}>
                        {col.name}
                      </td>
                      <td style={{ padding: '10px 14px' }}>
                        {col.key === 'PK' ? (
                          <span className="badge badge-low" style={{ fontSize: '10px' }}>PRIMARY KEY</span>
                        ) : col.key === 'FK' ? (
                          <span className="badge badge-primary" style={{ fontSize: '10px' }}>FOREIGN KEY</span>
                        ) : col.key === 'Target' ? (
                          <span className="badge badge-high" style={{ fontSize: '10px' }}>TARGET</span>
                        ) : (
                          <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Attribute</span>
                        )}
                      </td>
                      <td style={{ padding: '10px 14px', color: 'var(--accent-primary)', fontSize: '12px' }}>
                        {col.type}
                      </td>
                      <td style={{ padding: '10px 14px', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                        {col.desc}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

          </div>

        </div>

      </div>
    </div>
  );
}
