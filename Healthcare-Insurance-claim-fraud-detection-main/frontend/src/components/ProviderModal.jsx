import React, { useState, useEffect } from 'react';
import { 
  X, 
  ShieldAlert, 
  CheckCircle2, 
  AlertTriangle, 
  FileText, 
  DollarSign, 
  Users, 
  Building2, 
  Activity,
  Layers,
  ArrowUpRight,
  TrendingUp,
  Stethoscope,
  Info
} from 'lucide-react';

export default function ProviderModal({ 
  providerId, 
  analysisId, 
  onClose,
  backendUrl 
}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [claimFilter, setClaimFilter] = useState('all'); // 'all', 'inpatient', 'outpatient', 'anomalies'

  useEffect(() => {
    if (!providerId) return;

    const fetchDetails = async () => {
      setLoading(true);
      setError(null);
      try {
        const query = analysisId ? `?analysis_id=${analysisId}` : '';
        const res = await fetch(`${backendUrl}/provider/${providerId}${query}`);
        if (!res.ok) {
          const errJson = await res.json();
          throw new Error(errJson.detail || 'Failed to load provider details.');
        }
        const json = await res.json();
        setData(json);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDetails();
  }, [providerId, analysisId, backendUrl]);

  if (!providerId) return null;

  const filteredClaims = (data?.claims?.recent || []).filter(claim => {
    if (claimFilter === 'inpatient') return claim.ClaimType?.toLowerCase() === 'inpatient';
    if (claimFilter === 'outpatient') return claim.ClaimType?.toLowerCase() === 'outpatient';
    if (claimFilter === 'anomalies') return claim.AnomalyLevel === 'High' || claim.AnomalyLevel === 'Medium';
    return true;
  });

  const riskScore = data?.provider?.risk_score ?? 0;
  const isHighRisk = riskScore >= 75;
  const isMediumRisk = riskScore >= 40 && riskScore < 75;
  const riskColor = isHighRisk ? 'var(--risk-high)' : isMediumRisk ? 'var(--risk-medium)' : 'var(--risk-low)';

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 100,
      backgroundColor: 'rgba(0, 0, 0, 0.75)',
      backdropFilter: 'blur(8px)',
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
          maxWidth: '1000px',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          border: '1px solid var(--border-highlight)',
          boxShadow: '0 25px 60px -15px rgba(0, 0, 0, 0.7)'
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
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{
              width: '42px',
              height: '42px',
              borderRadius: '12px',
              background: 'rgba(99, 102, 241, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Building2 size={22} color="var(--accent-primary)" />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <h2 style={{ fontSize: '20px', fontWeight: '800', fontFamily: 'var(--font-mono)' }}>
                  {providerId}
                </h2>
                {data && (
                  <span className={`badge ${isHighRisk ? 'badge-high' : isMediumRisk ? 'badge-medium' : 'badge-low'}`}>
                    {isHighRisk ? <AlertTriangle size={12} /> : <CheckCircle2 size={12} />}
                    {data.provider.status}
                  </span>
                )}
              </div>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: '2px 0 0' }}>
                Comprehensive Provider Risk Profile & Anomaly Audit
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="btn-secondary"
            style={{ padding: '8px', borderRadius: '8px' }}
            title="Close modal"
          >
            <X size={18} />
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '28px', overflowY: 'auto', flex: 1 }}>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '60px 0' }}>
              <div className="spin-slow" style={{
                width: '36px',
                height: '36px',
                border: '3px solid var(--accent-primary)',
                borderTopColor: 'transparent',
                borderRadius: '50%',
                margin: '0 auto 16px'
              }} />
              <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                Retrieving provider statistics and scoring claim anomalies...
              </p>
            </div>
          ) : error ? (
            <div style={{
              padding: '20px',
              borderRadius: '12px',
              background: 'var(--risk-high-bg)',
              color: 'var(--risk-high)',
              fontSize: '14px'
            }}>
              {error}
            </div>
          ) : data ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              
              {/* Top Summary Banner Grid */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '16px'
              }}>
                {/* Risk Score Card */}
                <div className="glass-card" style={{ padding: '18px', borderLeft: `4px solid ${riskColor}` }}>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
                    Fraud Probability Score
                  </div>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                    <span style={{ fontSize: '28px', fontWeight: '800', color: riskColor }}>
                      {riskScore}%
                    </span>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                      ({data.provider.risk_level} Risk)
                    </span>
                  </div>
                  <div className="risk-meter" style={{ marginTop: '10px' }}>
                    <div 
                      className="risk-meter-fill" 
                      style={{ width: `${riskScore}%`, backgroundColor: riskColor }} 
                    />
                  </div>
                </div>

                {/* Total Reimbursement */}
                <div className="glass-card" style={{ padding: '18px' }}>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
                    Total Reimbursement
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: '800', color: 'var(--text-primary)' }}>
                    ${Number(data.statistics?.total_reimbursement || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    Avg: ${Number(data.statistics?.avg_claim_amount || 0).toFixed(0)} / claim
                  </div>
                </div>

                {/* Claim Counts */}
                <div className="glass-card" style={{ padding: '18px' }}>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
                    Claim Distribution
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: '800', color: 'var(--text-primary)' }}>
                    {data.statistics?.total_claims || 0}
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    {data.statistics?.inpatient_claims || 0} Inpatient · {data.statistics?.outpatient_claims || 0} Outpatient
                  </div>
                </div>

                {/* Claim Anomalies */}
                <div className="glass-card" style={{ padding: '18px' }}>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
                    Anomalous Claims
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: '800', color: data.claim_anomalies?.total_anomalous > 0 ? 'var(--risk-high)' : 'var(--risk-low)' }}>
                    {data.claim_anomalies?.total_anomalous || 0}
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    {data.claim_anomalies?.high || 0} High · {data.claim_anomalies?.medium || 0} Medium
                  </div>
                </div>
              </div>

              {/* AI Why Flagged Root Cause Analysis */}
              <div className="glass-card" style={{ padding: '24px', border: isHighRisk ? '1px solid rgba(244, 63, 94, 0.4)' : '1px solid var(--border-color)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Info size={18} color="var(--accent-cyan)" />
                    <h3 style={{ fontSize: '17px', fontWeight: '800' }}>
                      Explainable Fraud Risk Assessment & Contributing Factors
                    </h3>
                  </div>
                  <span className={`badge ${isHighRisk ? 'badge-high' : isMediumRisk ? 'badge-medium' : 'badge-low'}`}>
                    Risk Score: {riskScore}/100 ({data.provider.risk_level})
                  </span>
                </div>

                <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '16px' }}>
                  {data.why_flagged?.message}
                </p>

                {/* Plain-English Bulleted Reasons */}
                {data.why_flagged?.human_readable_reasons && data.why_flagged.human_readable_reasons.length > 0 && (
                  <div style={{ marginBottom: '20px', background: 'rgba(0, 0, 0, 0.2)', padding: '16px 20px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                    <div style={{ fontSize: '12px', fontWeight: '700', textTransform: 'uppercase', color: 'var(--text-primary)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <CheckCircle2 size={15} color="#10b981" />
                      Key Identified Fraud Patterns & Peer Anomalies:
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {data.why_flagged.human_readable_reasons.map((reason, idx) => (
                        <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', fontSize: '13px', color: '#e2e8f0', lineHeight: '1.5' }}>
                          <span style={{ color: '#10b981', fontWeight: 'bold' }}>✓</span>
                          <span>{reason}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Risk Factors Badges */}
                {data.why_flagged?.risk_factors && data.why_flagged.risk_factors.length > 0 ? (
                  <div>
                    <div style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '10px' }}>
                      Top ML Feature Attributions (SHAP Contribution):
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                      {data.why_flagged.risk_factors.map((factor, idx) => {
                        const featName = typeof factor === 'object' ? factor.feature : factor;
                        const featVal = typeof factor === 'object' ? factor.value : '';
                        const contrib = typeof factor === 'object' && factor.contribution ? `(+${factor.contribution})` : '';
                        return (
                          <div 
                            key={idx}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                              padding: '6px 12px',
                              borderRadius: '8px',
                              background: 'rgba(244, 63, 94, 0.1)',
                              border: '1px solid rgba(244, 63, 94, 0.25)',
                              fontSize: '12px',
                              color: '#fda4af'
                            }}
                          >
                            <TrendingUp size={13} color="#f43f5e" />
                            <span style={{ fontWeight: '600' }}>{featName}</span>
                            {featVal !== '' && <span style={{ color: '#fff' }}>: {featVal}</span>}
                            {contrib && <span style={{ fontSize: '10px', color: '#fca5a5' }}>{contrib}</span>}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: '13px', color: 'var(--risk-low)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <CheckCircle2 size={15} />
                    No critical risk anomalies flagged for this provider.
                  </div>
                )}
              </div>

              {/* Claims Drill-down Table Section */}
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
                  <div>
                    <h3 style={{ fontSize: '16px', fontWeight: '700' }}>
                      Recent Claims & Behavioral Anomalies
                    </h3>
                    <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      Showing {filteredClaims.length} claims with anomaly scores
                    </p>
                  </div>

                  {/* Filter Tabs */}
                  <div style={{ display: 'flex', gap: '6px' }}>
                    {['all', 'inpatient', 'outpatient', 'anomalies'].map((filter) => (
                      <button
                        key={filter}
                        type="button"
                        onClick={() => setClaimFilter(filter)}
                        className="btn-secondary"
                        style={{
                          padding: '6px 12px',
                          fontSize: '12px',
                          textTransform: 'capitalize',
                          background: claimFilter === filter ? 'var(--accent-primary)' : 'rgba(255, 255, 255, 0.04)',
                          borderColor: claimFilter === filter ? 'var(--accent-primary)' : 'var(--border-color)',
                          color: claimFilter === filter ? '#fff' : 'var(--text-secondary)'
                        }}
                      >
                        {filter === 'anomalies' ? '🚨 Anomalies Only' : filter}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="glass-card" style={{ overflowX: 'auto', borderRadius: '12px' }}>
                  <table className="custom-table">
                    <thead>
                      <tr>
                        <th>Claim ID</th>
                        <th>Type</th>
                        <th>Claim Amount</th>
                        <th>Duration</th>
                        <th>Diagnosis Count</th>
                        <th>Anomaly Score</th>
                        <th>Anomaly Reason</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredClaims.length > 0 ? (
                        filteredClaims.map((claim, idx) => {
                          const isClaimAnomaly = claim.AnomalyLevel === 'High' || claim.AnomalyLevel === 'Medium';
                          return (
                            <tr key={idx} style={{ background: isClaimAnomaly ? 'rgba(244, 63, 94, 0.04)' : 'transparent' }}>
                              <td style={{ fontFamily: 'var(--font-mono)', fontWeight: '600', color: 'var(--accent-cyan)' }}>
                                {claim.ClaimID}
                              </td>
                              <td>
                                <span className={`badge ${claim.ClaimType?.toLowerCase() === 'inpatient' ? 'badge-neutral' : 'badge-neutral'}`}>
                                  {claim.ClaimType}
                                </span>
                              </td>
                              <td style={{ fontWeight: '600' }}>
                                ${Number(claim.ClaimAmount || 0).toLocaleString()}
                              </td>
                              <td>
                                {claim.ClaimDuration ? `${claim.ClaimDuration} days` : 'N/A'}
                              </td>
                              <td>
                                {claim.DiagnosisCount || 0} codes
                              </td>
                              <td>
                                <span className={`badge ${claim.AnomalyLevel === 'High' ? 'badge-high' : claim.AnomalyLevel === 'Medium' ? 'badge-medium' : 'badge-low'}`}>
                                  {claim.AnomalyScore ? `${Math.round(claim.AnomalyScore * 100)}%` : 'Normal'}
                                </span>
                              </td>
                              <td style={{ fontSize: '12px', color: isClaimAnomaly ? '#fda4af' : 'var(--text-muted)' }}>
                                {Array.isArray(claim.AnomalyReasons) ? claim.AnomalyReasons.join(', ') : (claim.AnomalyReasons || 'Standard claim behavior')}
                              </td>
                            </tr>
                          );
                        })
                      ) : (
                        <tr>
                          <td colSpan={7} style={{ textAlign: 'center', padding: '30px', color: 'var(--text-muted)' }}>
                            No claims matching the selected filter.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
