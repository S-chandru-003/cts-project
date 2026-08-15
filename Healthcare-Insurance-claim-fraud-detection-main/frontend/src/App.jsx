import React, { useState, useEffect } from 'react';
import confetti from 'canvas-confetti';
import Navbar from './components/Navbar';
import UploadSection from './components/UploadSection';
import Dashboard from './components/Dashboard';
import ProviderModal from './components/ProviderModal';
import DataDictionaryModal from './components/DataDictionaryModal';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';

export default function App() {
  const [backendOnline, setBackendOnline] = useState(false);
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' or 'dashboard'
  const [analysisData, setAnalysisData] = useState(null);
  const [selectedProviderId, setSelectedProviderId] = useState(null);
  const [showDataDictionary, setShowDataDictionary] = useState(false);
  
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('');
  const [error, setError] = useState(null);

  // Check backend health on mount and periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/health`, { method: 'GET' });
        if (res.ok) {
          setBackendOnline(true);
        } else {
          setBackendOnline(false);
        }
      } catch {
        setBackendOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Handler for uploading files and analyzing
  const handleAnalyze = async (uploadPayload) => {
    setLoading(true);
    setError(null);
    setLoadingStage('Ingesting & Validating Datasets...');

    try {
      const formData = new FormData();

      if (uploadPayload.type === 'files') {
        uploadPayload.files.forEach((file) => {
          if (file) {
            formData.append('dataset_files', file, file.name);
          }
        });
      } else if (uploadPayload.type === 'zip') {
        formData.append('zip_file', uploadPayload.file);
      }

      // Simulated stage updates for smooth UX
      setTimeout(() => setLoadingStage('Engineering 120+ Provider & Claim Behavioral Features...'), 1200);
      setTimeout(() => setLoadingStage('Running CatBoost ML Fraud Inference Pipeline...'), 2800);
      setTimeout(() => setLoadingStage('Scoring Claim Anomalies & Building Geographic Visualizations...'), 4200);

      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Analysis failed. Please ensure the 4 CSV files are valid.');
      }

      setAnalysisData(data);
      setActiveTab('dashboard');

      // Trigger celebratory confetti
      confetti({
        particleCount: 80,
        spread: 70,
        origin: { y: 0.6 }
      });

    } catch (err) {
      setError(err.message || 'Could not complete fraud analysis.');
    } finally {
      setLoading(false);
      setLoadingStage('');
    }
  };

  // Handler for 1-click sample dataset demo
  const handleLoadSample = async () => {
    setLoading(true);
    setError(null);
    setLoadingStage('Loading Sample Healthcare Dataset...');

    try {
      setTimeout(() => setLoadingStage('Extracting Features & Running CatBoost Model...'), 1000);

      const response = await fetch(`${BACKEND_URL}/predict/sample`, {
        method: 'POST',
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to load sample dataset.');
      }

      setAnalysisData(data);
      setActiveTab('dashboard');

      confetti({
        particleCount: 80,
        spread: 70,
        origin: { y: 0.6 }
      });

    } catch (err) {
      setError(err.message || 'Could not load sample data. Ensure backend is running.');
    } finally {
      setLoading(false);
      setLoadingStage('');
    }
  };

  // Reset analysis
  const handleReset = () => {
    setAnalysisData(null);
    setSelectedProviderId(null);
    setActiveTab('upload');
    setError(null);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* Top Navigation */}
      <Navbar
        backendOnline={backendOnline}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        hasResults={Boolean(analysisData)}
        onLoadSample={handleLoadSample}
        onReset={handleReset}
        onOpenDataDictionary={() => setShowDataDictionary(true)}
        loading={loading}
      />

      {/* Main Content Area */}
      <main style={{ flex: 1 }}>
        {activeTab === 'upload' ? (
          <UploadSection
            onAnalyze={handleAnalyze}
            loading={loading}
            loadingStage={loadingStage}
            onLoadSample={handleLoadSample}
            error={error}
          />
        ) : (
          <Dashboard
            analysisData={analysisData}
            onSelectProvider={(id) => setSelectedProviderId(id)}
            onReUpload={() => setActiveTab('upload')}
          />
        )}
      </main>

      {/* Provider Details Deep Dive Modal */}
      {selectedProviderId && (
        <ProviderModal
          providerId={selectedProviderId}
          analysisId={analysisData?.analysis_id}
          onClose={() => setSelectedProviderId(null)}
          backendUrl={BACKEND_URL}
        />
      )}

      {/* Interactive Data Dictionary & Schema Modal */}
      {showDataDictionary && (
        <DataDictionaryModal
          onClose={() => setShowDataDictionary(false)}
        />
      )}

      {/* Footer */}
      <footer style={{
        marginTop: 'auto',
        borderTop: '1px solid var(--border-color)',
        padding: '20px 24px',
        textAlign: 'center',
        fontSize: '13px',
        color: 'var(--text-muted)'
      }}>
        <div>
          Healthcare Insurance Claim Fraud Detection & Behavioral Intelligence · Powered by CatBoost ML
        </div>
      </footer>

    </div>
  );
}

