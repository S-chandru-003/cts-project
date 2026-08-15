import React from 'react';
import { ShieldAlert, Activity, Sparkles, RefreshCw, BarChart2, UploadCloud, Database } from 'lucide-react';

export default function Navbar({ 
  backendOnline, 
  activeTab, 
  setActiveTab, 
  hasResults, 
  onLoadSample, 
  onReset,
  onOpenDataDictionary,
  loading 
}) {
  return (
    <header style={{
      position: 'sticky',
      top: 0,
      zIndex: 50,
      backgroundColor: 'rgba(11, 15, 25, 0.85)',
      backdropFilter: 'blur(16px)',
      borderBottom: '1px solid var(--border-color)',
      padding: '16px 24px'
    }}>
      <div style={{
        maxWidth: '1440px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        {/* Brand Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '42px',
            height: '42px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 16px rgba(99, 102, 241, 0.4)'
          }}>
            <ShieldAlert size={24} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '18px', fontWeight: '800', letterSpacing: '-0.03em' }}>
                HealthGuard <span className="gradient-text">AI</span>
              </span>
              <span className="badge badge-neutral" style={{ fontSize: '10px', padding: '2px 8px' }}>
                v2.0 CatBoost
              </span>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0 }}>
              Healthcare Provider Fraud Detection & Claim Intelligence
            </p>
          </div>
        </div>

        {/* View Switcher Tabs */}
        <div style={{
          display: 'flex',
          background: 'rgba(255, 255, 255, 0.04)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '4px',
          gap: '4px'
        }}>
          <button
            onClick={() => setActiveTab('upload')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 16px',
              borderRadius: '8px',
              border: 'none',
              fontSize: '13px',
              fontWeight: '600',
              cursor: 'pointer',
              background: activeTab === 'upload' ? 'var(--accent-primary)' : 'transparent',
              color: activeTab === 'upload' ? '#fff' : 'var(--text-secondary)',
              transition: 'all 0.2s ease'
            }}
          >
            <UploadCloud size={16} />
            Data Upload
          </button>

          <button
            onClick={() => setActiveTab('dashboard')}
            disabled={!hasResults}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 16px',
              borderRadius: '8px',
              border: 'none',
              fontSize: '13px',
              fontWeight: '600',
              cursor: hasResults ? 'pointer' : 'not-allowed',
              opacity: hasResults ? 1 : 0.45,
              background: activeTab === 'dashboard' ? 'var(--accent-primary)' : 'transparent',
              color: activeTab === 'dashboard' ? '#fff' : 'var(--text-secondary)',
              transition: 'all 0.2s ease'
            }}
          >
            <BarChart2 size={16} />
            Fraud Dashboard
            {hasResults && (
              <span style={{
                width: '7px',
                height: '7px',
                borderRadius: '50%',
                backgroundColor: '#10b981'
              }} />
            )}
          </button>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Data Dictionary Button */}
          <button
            onClick={onOpenDataDictionary}
            className="btn-secondary"
            title="Inspect schema, relationships, and data dictionary"
            style={{ fontSize: '13px', padding: '8px 14px' }}
          >
            <Database size={15} color="var(--accent-primary)" />
            Data Dictionary & Schema
          </button>

          {/* Quick Demo Sample Data Button */}
          <button
            onClick={onLoadSample}
            disabled={loading}
            className="btn-secondary"
            title="Analyze the included test dataset with 1 click"
            style={{ fontSize: '13px', padding: '8px 14px' }}
          >
            <Sparkles size={15} color="#06b6d4" />
            Load Sample Dataset
          </button>

          {/* Reset button if has results */}
          {hasResults && (
            <button
              onClick={onReset}
              className="btn-secondary"
              title="Reset analysis and upload new files"
              style={{ fontSize: '13px', padding: '8px 12px' }}
            >
              <RefreshCw size={14} />
              Reset
            </button>
          )}

          {/* Backend Status Badge */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            borderRadius: '10px',
            background: backendOnline ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
            border: `1px solid ${backendOnline ? 'rgba(16, 185, 129, 0.2)' : 'rgba(244, 63, 94, 0.2)'}`,
            fontSize: '12px',
            fontWeight: '500',
            color: backendOnline ? 'var(--risk-low)' : 'var(--risk-high)'
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: backendOnline ? '#10b981' : '#f43f5e',
              boxShadow: backendOnline ? '0 0 8px #10b981' : '0 0 8px #f43f5e'
            }} />
            <span>{backendOnline ? 'API Connected' : 'API Offline'}</span>
          </div>
        </div>
      </div>
    </header>
  );
}

