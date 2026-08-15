import React, { useState } from 'react';
import { 
  Building2, 
  Users, 
  Hospital, 
  Stethoscope, 
  Upload, 
  FileCheck2, 
  X, 
  FileArchive, 
  Sparkles, 
  ArrowRight, 
  AlertCircle,
  Cpu,
  CheckCircle2,
  FolderOpen
} from 'lucide-react';

export default function UploadSection({ 
  onAnalyze, 
  loading, 
  loadingStage, 
  onLoadSample,
  error 
}) {
  const [mode, setMode] = useState('four-files'); // 'four-files' or 'zip'
  const [providerFile, setProviderFile] = useState(null);
  const [beneficiaryFile, setBeneficiaryFile] = useState(null);
  const [inpatientFile, setInpatientFile] = useState(null);
  const [outpatientFile, setOutpatientFile] = useState(null);
  const [zipFile, setZipFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  // Format file size
  const formatSize = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Helper to auto-categorize dropped files by name
  const handleAutoAssignFiles = (fileList) => {
    const files = Array.from(fileList);
    
    // Check if zip
    const zip = files.find(f => f.name.toLowerCase().endsWith('.zip'));
    if (zip) {
      setZipFile(zip);
      setMode('zip');
      return;
    }

    files.forEach(file => {
      const name = file.name.toLowerCase();
      if (name.includes('provider')) {
        setProviderFile(file);
      } else if (name.includes('bene')) {
        setBeneficiaryFile(file);
      } else if (name.includes('inpatient')) {
        setInpatientFile(file);
      } else if (name.includes('outpatient')) {
        setOutpatientFile(file);
      } else if (name.endsWith('.csv')) {
        // Fallback slot assignment
        if (!providerFile && !name.includes('inpatient') && !name.includes('outpatient') && !name.includes('bene')) {
          setProviderFile(file);
        }
      }
    });
  };

  // Drag-and-drop container handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleAutoAssignFiles(e.dataTransfer.files);
    }
  };

  const allFourUploaded = providerFile && beneficiaryFile && inpatientFile && outpatientFile;
  const isReadyToAnalyze = mode === 'four-files' ? allFourUploaded : Boolean(zipFile);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isReadyToAnalyze) return;

    if (mode === 'four-files') {
      onAnalyze({
        type: 'files',
        files: [providerFile, beneficiaryFile, inpatientFile, outpatientFile]
      });
    } else {
      onAnalyze({
        type: 'zip',
        file: zipFile
      });
    }
  };

  return (
    <div className="animate-fade-in" style={{ maxWidth: '1200px', margin: '30px auto', padding: '0 20px' }}>
      
      {/* Header Banner */}
      <div style={{ textAlign: 'center', marginBottom: '36px' }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 16px',
          borderRadius: '9999px',
          background: 'rgba(99, 102, 241, 0.1)',
          border: '1px solid rgba(99, 102, 241, 0.25)',
          color: '#818cf8',
          fontSize: '13px',
          fontWeight: '600',
          marginBottom: '16px'
        }}>
          <Cpu size={15} />
          CatBoost Gradient Boosting Fraud Detection Engine
        </div>

        <h1 style={{ fontSize: '38px', fontWeight: '800', lineHeight: 1.2, marginBottom: '12px' }}>
          Healthcare Insurance <span className="gradient-text">Claim Fraud Analysis</span>
        </h1>
        <p style={{ fontSize: '16px', color: 'var(--text-secondary)', maxWidth: '680px', margin: '0 auto', lineHeight: 1.6 }}>
          Upload your healthcare claims data across the 4 core datasets. Our machine learning pipeline performs feature engineering, behavioral anomaly detection, and risk scoring in seconds.
        </p>
      </div>

      {/* Mode Switcher Pill */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        marginBottom: '28px'
      }}>
        <div style={{
          display: 'inline-flex',
          background: 'rgba(255, 255, 255, 0.04)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '4px',
          gap: '4px'
        }}>
          <button
            type="button"
            onClick={() => setMode('four-files')}
            style={{
              padding: '8px 20px',
              borderRadius: '8px',
              border: 'none',
              fontSize: '14px',
              fontWeight: '600',
              cursor: 'pointer',
              background: mode === 'four-files' ? 'var(--accent-primary)' : 'transparent',
              color: mode === 'four-files' ? '#fff' : 'var(--text-secondary)',
              transition: 'all 0.2s'
            }}
          >
            4 Individual CSV Files
          </button>
          <button
            type="button"
            onClick={() => setMode('zip')}
            style={{
              padding: '8px 20px',
              borderRadius: '8px',
              border: 'none',
              fontSize: '14px',
              fontWeight: '600',
              cursor: 'pointer',
              background: mode === 'zip' ? 'var(--accent-primary)' : 'transparent',
              color: mode === 'zip' ? '#fff' : 'var(--text-secondary)',
              transition: 'all 0.2s'
            }}
          >
            Single ZIP Archive (.zip)
          </button>
        </div>
      </div>

      {/* Error Alert if any */}
      {error && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: '16px 20px',
          borderRadius: '12px',
          background: 'var(--risk-high-bg)',
          border: '1px solid var(--risk-high-border)',
          color: 'var(--risk-high)',
          marginBottom: '28px',
          fontSize: '14px'
        }}>
          <AlertCircle size={20} style={{ flexShrink: 0 }} />
          <div style={{ flex: 1 }}>{error}</div>
        </div>
      )}

      {/* 4 Files Upload Grid */}
      {mode === 'four-files' ? (
        <div 
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          style={{
            position: 'relative',
            borderRadius: '20px',
            border: dragActive ? '2px dashed var(--accent-primary)' : '1px solid transparent',
            padding: '4px',
            transition: 'border-color 0.2s'
          }}
        >
          {dragActive && (
            <div style={{
              position: 'absolute',
              inset: 0,
              background: 'rgba(99, 102, 241, 0.15)',
              backdropFilter: 'blur(4px)',
              borderRadius: '20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 20,
              fontSize: '18px',
              fontWeight: '600',
              color: '#818cf8'
            }}>
              Drop all 4 CSV files here to auto-assign!
            </div>
          )}

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
            gap: '20px',
            marginBottom: '32px'
          }}>
            {/* Slot 1: Provider Dataset */}
            <FileUploadCard
              title="1. Provider Dataset"
              subtitle="PROVIDERS.csv / Test.csv"
              description="Provider identifiers and known target labels"
              icon={<Building2 size={24} color="#6366f1" />}
              file={providerFile}
              onSelect={(f) => setProviderFile(f)}
              onRemove={() => setProviderFile(null)}
              formatSize={formatSize}
              accept=".csv"
            />

            {/* Slot 2: Beneficiary Dataset */}
            <FileUploadCard
              title="2. Beneficiary Dataset"
              subtitle="BENEFICIARY.csv"
              description="Patient demographics, chronic condition flags, deductibles"
              icon={<Users size={24} color="#06b6d4" />}
              file={beneficiaryFile}
              onSelect={(f) => setBeneficiaryFile(f)}
              onRemove={() => setBeneficiaryFile(null)}
              formatSize={formatSize}
              accept=".csv"
            />

            {/* Slot 3: Inpatient Claims */}
            <FileUploadCard
              title="3. Inpatient Claims"
              subtitle="INPATIENT.csv"
              description="Hospital admissions, length of stay, procedure codes"
              icon={<Hospital size={24} color="#8b5cf6" />}
              file={inpatientFile}
              onSelect={(f) => setInpatientFile(f)}
              onRemove={() => setInpatientFile(null)}
              formatSize={formatSize}
              accept=".csv"
            />

            {/* Slot 4: Outpatient Claims */}
            <FileUploadCard
              title="4. Outpatient Claims"
              subtitle="OUTPATIENT.csv"
              description="Doctor visits, diagnostic codes, reimbursement charges"
              icon={<Stethoscope size={24} color="#ec4899" />}
              file={outpatientFile}
              onSelect={(f) => setOutpatientFile(f)}
              onRemove={() => setOutpatientFile(null)}
              formatSize={formatSize}
              accept=".csv"
            />
          </div>
        </div>
      ) : (
        /* ZIP Archive Upload Card */
        <div style={{ marginBottom: '32px' }}>
          <div className="glass-card" style={{
            padding: '40px 24px',
            textAlign: 'center',
            border: '2px dashed var(--border-color)',
            borderRadius: '16px'
          }}>
            <div style={{
              width: '64px',
              height: '64px',
              borderRadius: '16px',
              background: 'rgba(99, 102, 241, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px'
            }}>
              <FileArchive size={32} color="#818cf8" />
            </div>

            <h3 style={{ fontSize: '18px', fontWeight: '700', marginBottom: '8px' }}>
              {zipFile ? zipFile.name : 'Upload ZIP Package'}
            </h3>
            <p style={{ fontSize: '14px', color: 'var(--text-secondary)', maxWidth: '440px', margin: '0 auto 20px' }}>
              {zipFile 
                ? `Size: ${formatSize(zipFile.size)} — Ready to extract and analyze.`
                : 'Upload a compressed .zip file containing Provider, Beneficiary, Inpatient, and Outpatient CSV datasets.'}
            </p>

            <div style={{ display: 'flex', justifyContent: 'center', gap: '12px' }}>
              <label className="btn-secondary" style={{ cursor: 'pointer' }}>
                <Upload size={16} />
                {zipFile ? 'Choose Different ZIP' : 'Browse ZIP File'}
                <input 
                  type="file" 
                  accept=".zip" 
                  style={{ display: 'none' }} 
                  onChange={(e) => e.target.files?.[0] && setZipFile(e.target.files[0])} 
                />
              </label>

              {zipFile && (
                <button
                  type="button"
                  onClick={() => setZipFile(null)}
                  className="btn-secondary"
                  style={{ color: 'var(--risk-high)' }}
                >
                  <X size={16} /> Remove
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Action Footer Bar */}
      <div className="glass-card" style={{
        padding: '24px 32px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '20px'
      }}>
        {/* Upload Summary Status */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: isReadyToAnalyze ? '#10b981' : '#6b7280'
            }} />
            <span style={{ fontSize: '15px', fontWeight: '600' }}>
              {mode === 'four-files' 
                ? `${[providerFile, beneficiaryFile, inpatientFile, outpatientFile].filter(Boolean).length} of 4 Datasets Ready`
                : (zipFile ? 'ZIP Archive Ready' : 'No ZIP Selected')}
            </span>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '4px 0 0' }}>
            {isReadyToAnalyze 
              ? 'All required datasets are configured. Click View Results to start inference.' 
              : 'Please load all four CSV datasets or upload a ZIP file to proceed.'}
          </p>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <button
            type="button"
            onClick={onLoadSample}
            disabled={loading}
            className="btn-secondary"
            style={{ padding: '12px 20px', fontSize: '14px' }}
          >
            <Sparkles size={16} color="#06b6d4" />
            Quick Demo with Sample Data
          </button>

          <button
            type="button"
            onClick={handleSubmit}
            disabled={!isReadyToAnalyze || loading}
            className="btn-primary"
            style={{ padding: '12px 28px', fontSize: '15px' }}
          >
            {loading ? (
              <>
                <div className="spin-slow" style={{ width: '16px', height: '16px', border: '2px solid #fff', borderTopColor: 'transparent', borderRadius: '50%' }} />
                <span>Analyzing Dataset...</span>
              </>
            ) : (
              <>
                <span>⚡ View Results & Dashboard</span>
                <ArrowRight size={18} />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Loading Progress State Modal / Banner */}
      {loading && (
        <div className="glass-card animate-fade-in" style={{
          marginTop: '24px',
          padding: '28px',
          border: '1px solid var(--accent-primary)',
          boxShadow: '0 10px 40px -10px rgba(99, 102, 241, 0.3)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '20px' }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '50%',
              border: '3px solid rgba(99, 102, 241, 0.2)',
              borderTopColor: 'var(--accent-primary)',
              animation: 'spin 1s linear infinite'
            }} />
            <div>
              <h4 style={{ fontSize: '16px', fontWeight: '700', margin: 0 }}>
                {loadingStage || 'Processing Healthcare Datasets...'}
              </h4>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', margin: '4px 0 0' }}>
                Executing automated data engineering, CatBoost inference, and anomaly isolation.
              </p>
            </div>
          </div>

          {/* Stepper Progress */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '12px'
          }}>
            <ProgressStep 
              label="1. Ingest & Validate CSVs" 
              active={loadingStage?.includes('Ingest') || loadingStage?.includes('Parsing')} 
              completed={true} 
            />
            <ProgressStep 
              label="2. Engineer 50+ Features" 
              active={loadingStage?.includes('Engineer') || loadingStage?.includes('Feature')} 
              completed={loadingStage?.includes('CatBoost') || loadingStage?.includes('Dashboard')} 
            />
            <ProgressStep 
              label="3. CatBoost ML Scoring" 
              active={loadingStage?.includes('CatBoost') || loadingStage?.includes('ML')} 
              completed={loadingStage?.includes('Dashboard')} 
            />
            <ProgressStep 
              label="4. Anomaly Analytics" 
              active={loadingStage?.includes('Dashboard') || loadingStage?.includes('Anomaly')} 
              completed={false} 
            />
          </div>
        </div>
      )}

    </div>
  );
}

// Sub-component: File Upload Card
function FileUploadCard({
  title,
  subtitle,
  description,
  icon,
  file,
  onSelect,
  onRemove,
  formatSize,
  accept
}) {
  const isLoaded = Boolean(file);

  return (
    <div className="glass-card" style={{
      padding: '24px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      position: 'relative',
      border: isLoaded ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid var(--border-color)',
      background: isLoaded ? 'rgba(16, 185, 129, 0.03)' : 'var(--bg-card)'
    }}>
      <div>
        {/* Top Icon and Status */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
          <div style={{
            width: '46px',
            height: '46px',
            borderRadius: '12px',
            background: 'rgba(255, 255, 255, 0.04)',
            border: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            {icon}
          </div>

          {isLoaded ? (
            <span className="badge badge-low">
              <CheckCircle2 size={13} />
              Loaded
            </span>
          ) : (
            <span className="badge badge-neutral">
              Required
            </span>
          )}
        </div>

        {/* Titles */}
        <h3 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '4px' }}>
          {title}
        </h3>
        <div style={{
          fontSize: '12px',
          fontFamily: 'var(--font-mono)',
          color: 'var(--accent-cyan)',
          marginBottom: '8px'
        }}>
          {subtitle}
        </div>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.4, minHeight: '36px' }}>
          {description}
        </p>
      </div>

      {/* File Action Area */}
      <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid var(--border-color)' }}>
        {isLoaded ? (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'rgba(0, 0, 0, 0.25)',
            padding: '8px 12px',
            borderRadius: '8px'
          }}>
            <div style={{ overflow: 'hidden', marginRight: '8px' }}>
              <div style={{
                fontSize: '12px',
                fontWeight: '600',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                color: 'var(--text-primary)'
              }}>
                {file.name}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                {formatSize(file.size)}
              </div>
            </div>

            <button
              type="button"
              onClick={onRemove}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                padding: '4px',
                display: 'flex',
                alignItems: 'center'
              }}
              title="Remove file"
            >
              <X size={16} />
            </button>
          </div>
        ) : (
          <label style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            padding: '10px',
            borderRadius: '8px',
            background: 'rgba(255, 255, 255, 0.04)',
            border: '1px dashed rgba(255, 255, 255, 0.15)',
            cursor: 'pointer',
            fontSize: '13px',
            fontWeight: '600',
            color: 'var(--text-secondary)',
            transition: 'all 0.2s'
          }}>
            <Upload size={14} />
            Browse File
            <input
              type="file"
              accept={accept}
              style={{ display: 'none' }}
              onChange={(e) => e.target.files?.[0] && onSelect(e.target.files[0])}
            />
          </label>
        )}
      </div>
    </div>
  );
}

// Sub-component: Progress step indicator
function ProgressStep({ label, active, completed }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '10px 14px',
      borderRadius: '10px',
      background: active ? 'rgba(99, 102, 241, 0.15)' : 'rgba(255, 255, 255, 0.02)',
      border: `1px solid ${active ? 'var(--accent-primary)' : 'var(--border-color)'}`,
      fontSize: '12px',
      fontWeight: '600',
      color: active ? 'var(--accent-primary)' : completed ? 'var(--text-primary)' : 'var(--text-muted)'
    }}>
      {completed ? (
        <CheckCircle2 size={15} color="#10b981" />
      ) : active ? (
        <div style={{
          width: '12px',
          height: '12px',
          borderRadius: '50%',
          border: '2px solid var(--accent-primary)',
          borderTopColor: 'transparent',
          animation: 'spin 1s linear infinite'
        }} />
      ) : (
        <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: 'rgba(255, 255, 255, 0.1)' }} />
      )}
      <span>{label}</span>
    </div>
  );
}
