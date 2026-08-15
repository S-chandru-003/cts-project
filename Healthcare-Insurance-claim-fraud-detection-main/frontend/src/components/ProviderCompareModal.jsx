import React, { useState, useEffect } from 'react';
import { 
  X, 
  ArrowLeftRight, 
  Building2, 
  ShieldAlert, 
  ShieldCheck, 
  DollarSign, 
  Users, 
  FileText, 
  Layers, 
  TrendingUp, 
  TrendingDown,
  AlertTriangle,
  CheckCircle2,
  Sparkles,
  BarChart3
} from 'lucide-react';

export default function ProviderCompareModal({ 
  initialProviderId, 
  allProviders = [], 
  analysisId, 
  backendUrl, 
  onClose,
  onSelectProvider 
}) {
  const [provider1Id, setProvider1Id] = useState(initialProviderId || (allProviders[0]?.Provider || ''));
  const [provider2Id, setProvider2Id] = useState(
    allProviders.find(p => p.Provider !== initialProviderId)?.Provider || (allProviders[1]?.Provider || '')
  );
  const [compareData, setCompareData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!provider1Id) return;

    const fetchComparison = async () => {
      setLoading(true);
      setError(null);
      try {
        const queryParams = new URLSearchParams();
        queryParams.append('provider1', provider1Id);
        if (provider2Id) queryParams.append('provider2', provider2Id);
        if (analysisId) queryParams.append('analysis_id', analysisId);

        const res = await fetch(`${backendUrl}/compare?${queryParams.toString()}`);
        if (!res.ok) {
          const errJson = await res.json();
          throw new Error(errJson.detail || 'Could not fetch provider comparison.');
        }
        const data = await res.json();
        setCompareData(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchComparison();
  }, [provider1Id, provider2Id, analysisId, backendUrl]);

  const p1 = compareData?.provider1;
  const p2 = compareData?.provider2;
  const peer = compareData?.peer_benchmarks || {};

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
      padding: '20px',
      overflowY: 'auto'
    }}>
      <div 
        className="glass-card animate-fade-in" 
        style={{
          width: '100%',
          maxWidth: '1150px',
          maxHeight: '92vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          border: '1px solid var(--border-highlight)',
          boxShadow: '0 30px 70px -15px rgba(0, 0, 0, 0.8)'
        }}
      >
        {/* Header */}
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
              <ArrowLeftRight size={22} color="var(--accent-primary)" />
            </div>
            <div>
              <h2 style={{ fontSize: '20px', fontWeight: '800' }}>
                Side-by-Side <span className="gradient-text">Provider Comparison & Benchmarking</span>
              </h2>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: '2px 0 0' }}>
                Compare provider risk profiles, claim volumes, and reimbursement distributions against peers
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

        {/* Selector Bar */}
        <div style={{
          padding: '16px 28px',
          background: 'rgba(0, 0, 0, 0.2)',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '16px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1 }}>
            <span style={{ fontSize: '13px', fontWeight: '700', color: 'var(--accent-primary)' }}>Provider A:</span>
            <select
              value={provider1Id}
              onChange={(e) => setProvider1Id(e.target.value)}
              className="search-input"
              style={{ padding: '8px 12px', minWidth: '180px', fontFamily: 'var(--font-mono)' }}
            >
              {allProviders.map(p => (
                <option key={p.Provider} value={p.Provider}>
                  {p.Provider} ({Math.round(p.FraudProbability)}% Risk - {p.Prediction})
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}>
            <span style={{ fontSize: '12px', fontWeight: '800', textTransform: 'uppercase' }}>VS</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1, justifyContent: 'flex-end' }}>
            <span style={{ fontSize: '13px', fontWeight: '700', color: '#10b981' }}>Provider B:</span>
            <select
              value={provider2Id}
              onChange={(e) => setProvider2Id(e.target.value)}
              className="search-input"
              style={{ padding: '8px 12px', minWidth: '180px', fontFamily: 'var(--font-mono)' }}
            >
              {allProviders.map(p => (
                <option key={p.Provider} value={p.Provider}>
                  {p.Provider} ({Math.round(p.FraudProbability)}% Risk - {p.Prediction})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Content Body */}
        <div style={{ padding: '24px 28px', overflowY: 'auto', flex: 1 }}>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '60px' }}>
              <div className="animate-spin" style={{ width: '32px', height: '32px', border: '3px solid var(--accent-primary)', borderTopColor: 'transparent', borderRadius: '50%', margin: '0 auto 16px' }} />
              <p style={{ color: 'var(--text-muted)' }}>Calculating comparison metrics & peer deltas...</p>
            </div>
          ) : error ? (
            <div style={{ padding: '20px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid #ef4444', borderRadius: '8px', color: '#ef4444' }}>
              {error}
            </div>
          ) : p1 && p2 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              
              {/* Top Summary Comparison Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 180px 1fr', gap: '16px', alignItems: 'stretch' }}>
                
                {/* Provider 1 Card */}
                <div className="glass-card" style={{ padding: '20px', border: p1.risk_score >= 75 ? '1px solid #ef4444' : '1px solid var(--border-color)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <h3 style={{ fontSize: '20px', fontWeight: '800', fontFamily: 'var(--font-mono)', margin: 0 }}>{p1.provider_id}</h3>
                    <span className={`badge ${p1.risk_score >= 75 ? 'badge-high' : p1.risk_score >= 40 ? 'badge-medium' : 'badge-low'}`}>
                      {p1.status}
                    </span>
                  </div>
                  <div style={{ fontSize: '32px', fontWeight: '800', color: p1.risk_score >= 75 ? 'var(--risk-high)' : p1.risk_score >= 40 ? 'var(--risk-medium)' : 'var(--risk-low)' }}>
                    {p1.risk_score}%
                    <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 'normal', marginLeft: '6px' }}>Risk Score</span>
                  </div>
                  <button 
                    onClick={() => { onClose(); onSelectProvider(p1.provider_id); }}
                    className="btn-secondary" 
                    style={{ width: '100%', marginTop: '12px', fontSize: '12px', padding: '6px' }}
                  >
                    View Full Audit Profile
                  </button>
                </div>

                {/* Center Peer Benchmark Box */}
                <div style={{
                  padding: '16px',
                  borderRadius: '12px',
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: '1px dashed var(--border-color)',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  textAlign: 'center'
                }}>
                  <BarChart3 size={20} color="var(--text-muted)" />
                  <div style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: 'var(--text-muted)', marginTop: '4px' }}>
                    Peer Baseline
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: '1.4' }}>
                    Dataset Avg: <strong>${peer.average_claim_amount?.toLocaleString() || 0}</strong> / claim
                  </div>
                </div>

                {/* Provider 2 Card */}
                <div className="glass-card" style={{ padding: '20px', border: p2.risk_score >= 75 ? '1px solid #ef4444' : '1px solid var(--border-color)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <h3 style={{ fontSize: '20px', fontWeight: '800', fontFamily: 'var(--font-mono)', margin: 0 }}>{p2.provider_id}</h3>
                    <span className={`badge ${p2.risk_score >= 75 ? 'badge-high' : p2.risk_score >= 40 ? 'badge-medium' : 'badge-low'}`}>
                      {p2.status}
                    </span>
                  </div>
                  <div style={{ fontSize: '32px', fontWeight: '800', color: p2.risk_score >= 75 ? 'var(--risk-high)' : p2.risk_score >= 40 ? 'var(--risk-medium)' : 'var(--risk-low)' }}>
                    {p2.risk_score}%
                    <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 'normal', marginLeft: '6px' }}>Risk Score</span>
                  </div>
                  <button 
                    onClick={() => { onClose(); onSelectProvider(p2.provider_id); }}
                    className="btn-secondary" 
                    style={{ width: '100%', marginTop: '12px', fontSize: '12px', padding: '6px' }}
                  >
                    View Full Audit Profile
                  </button>
                </div>

              </div>

              {/* Metric Comparison Table */}
              <div className="table-container">
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-color)', background: 'rgba(255, 255, 255, 0.02)' }}>
                      <th style={{ padding: '12px 16px', textAlign: 'left', color: 'var(--text-muted)' }}>Behavioral & Financial Metric</th>
                      <th style={{ padding: '12px 16px', textAlign: 'center', color: 'var(--accent-primary)' }}>Provider A ({p1.provider_id})</th>
                      <th style={{ padding: '12px 16px', textAlign: 'center', color: 'var(--text-muted)' }}>Peer Benchmark</th>
                      <th style={{ padding: '12px 16px', textAlign: 'center', color: '#10b981' }}>Provider B ({p2.provider_id})</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '12px 16px', fontWeight: '600' }}>Total Reimbursement Paid</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: '700' }}>${p1.total_reimbursement?.toLocaleString()}</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', color: 'var(--text-muted)' }}>${peer.average_reimbursement_per_provider?.toLocaleString()} (Avg)</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: '700' }}>${p2.total_reimbursement?.toLocaleString()}</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '12px 16px', fontWeight: '600' }}>Average Claim Reimbursement</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', fontFamily: 'var(--font-mono)', color: p1.average_claim_amount > (peer.average_claim_amount || 0) * 1.5 ? '#ef4444' : 'inherit' }}>
                        ${p1.average_claim_amount?.toLocaleString()}
                      </td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', color: 'var(--text-muted)' }}>${peer.average_claim_amount?.toLocaleString()}</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', fontFamily: 'var(--font-mono)', color: p2.average_claim_amount > (peer.average_claim_amount || 0) * 1.5 ? '#ef4444' : 'inherit' }}>
                        ${p2.average_claim_amount?.toLocaleString()}
                      </td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '12px 16px', fontWeight: '600' }}>Total Claims Volume</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{p1.total_claims?.toLocaleString()}</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', color: 'var(--text-muted)' }}>{peer.average_claims_per_provider}</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{p2.total_claims?.toLocaleString()}</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '12px 16px', fontWeight: '600' }}>Inpatient vs Outpatient Ratio</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                        {p1.inpatient_ratio}% Inpatient ({p1.inpatient_claims} IP / {p1.outpatient_claims} OP)
                      </td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', color: 'var(--text-muted)' }}>{peer.inpatient_ratio}% IP</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                        {p2.inpatient_ratio}% Inpatient ({p2.inpatient_claims} IP / {p2.outpatient_claims} OP)
                      </td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '12px 16px', fontWeight: '600' }}>Unique Beneficiaries Served</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{p1.total_beneficiaries?.toLocaleString()}</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', color: 'var(--text-muted)' }}>-</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{p2.total_beneficiaries?.toLocaleString()}</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '12px 16px', fontWeight: '600' }}>Claims Per Beneficiary (Revisit Rate)</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{p1.claims_per_beneficiary}</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', color: 'var(--text-muted)' }}>{peer.average_claims_per_beneficiary}</td>
                      <td style={{ padding: '12px 16px', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{p2.claims_per_beneficiary}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

            </div>
          ) : null}
        </div>

      </div>
    </div>
  );
}
