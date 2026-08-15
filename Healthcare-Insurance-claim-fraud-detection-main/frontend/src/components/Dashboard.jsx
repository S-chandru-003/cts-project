import React, { useState, useMemo } from 'react';
import { 
  ShieldAlert, 
  ShieldCheck, 
  Users, 
  FileText, 
  DollarSign, 
  Building2, 
  Search, 
  Download, 
  Filter, 
  ArrowUpDown, 
  ChevronRight, 
  ExternalLink,
  UploadCloud,
  Layers,
  AlertTriangle,
  PieChart,
  BarChart3,
  TrendingDown,
  Sparkles,
  MapPin,
  ArrowLeftRight,
  Hospital,
  Activity
} from 'lucide-react';
import { 
  RiskDistributionChart, 
  FraudClassificationChart, 
  TopProvidersBarChart,
  GeographicStateBarChart,
  ClaimTypeBreakdownChart
} from './Charts';
import ProviderCompareModal from './ProviderCompareModal';

export default function Dashboard({ 
  analysisData, 
  onSelectProvider, 
  onReUpload 
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all'); // 'all', 'fraud', 'high_risk', 'medium_risk', 'low_risk'
  const [sortField, setSortField] = useState('FraudProbability');
  const [sortAsc, setSortAsc] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [compareInitialProvider, setCompareInitialProvider] = useState(null);
  const itemsPerPage = 12;

  const { summary, results, analysis_id, threshold, model, geographic_insights, claim_type_summary, peer_benchmarks } = analysisData || {};

  // Calculate financial risk exposure
  const totalReimbursement = summary?.total_reimbursement || 0;
  const fraudCount = summary?.potential_fraud || 0;
  const totalProviders = summary?.total_providers || 1;
  const fraudRate = summary?.fraud_percentage || (fraudCount / totalProviders * 100);

  // Filter & sort provider results
  const filteredProviders = useMemo(() => {
    if (!results || !Array.isArray(results)) return [];

    return results
      .filter((p) => {
        const matchesSearch = p.Provider?.toLowerCase().includes(searchTerm.toLowerCase().trim());
        if (!matchesSearch) return false;

        const prob = p.FraudProbability ?? 0;
        const isFraud = p.Prediction === 'Potential Fraud';

        if (filterType === 'fraud') return isFraud;
        if (filterType === 'high_risk') return prob >= 75;
        if (filterType === 'medium_risk') return prob >= 40 && prob < 75;
        if (filterType === 'low_risk') return prob < 40;
        return true;
      })
      .sort((a, b) => {
        let valA = a[sortField] ?? 0;
        let valB = b[sortField] ?? 0;
        if (typeof valA === 'string') valA = valA.toLowerCase();
        if (typeof valB === 'string') valB = valB.toLowerCase();

        if (valA < valB) return sortAsc ? -1 : 1;
        if (valA > valB) return sortAsc ? 1 : -1;
        return 0;
      });
  }, [results, searchTerm, filterType, sortField, sortAsc]);

  // Pagination
  const totalPages = Math.ceil(filteredProviders.length / itemsPerPage) || 1;
  const paginatedProviders = filteredProviders.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  // Sort toggle handler
  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  // Export CSV Handler
  const exportToCSV = () => {
    if (!results || results.length === 0) return;
    const headers = Object.keys(results[0]).join(',');
    const rows = results.map(r => Object.values(r).map(val => `"${val}"`).join(','));
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers, ...rows].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `fraud_analysis_${analysis_id || 'export'}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="animate-fade-in" style={{ maxWidth: '1440px', margin: '24px auto', padding: '0 24px 60px' }}>
      
      {/* Dashboard Top Header & Toolbar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px',
        marginBottom: '28px'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h1 style={{ fontSize: '28px', fontWeight: '800' }}>
              Healthcare Fraud <span className="gradient-text">Intelligence Dashboard</span>
            </h1>
            <span className="badge badge-neutral" style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
              ID: {analysis_id ? `${analysis_id.slice(0, 8)}...` : 'Active'}
            </span>
          </div>
          <p style={{ fontSize: '14px', color: 'var(--text-secondary)', margin: '4px 0 0' }}>
            Single-view executive summary, geographic insights, and provider-level anomaly scoring powered by CatBoost ML.
          </p>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={() => { setCompareInitialProvider(results?.[0]?.Provider); setShowCompareModal(true); }}
            className="btn-secondary"
            style={{ fontSize: '13px', padding: '8px 16px', color: 'var(--accent-primary)', borderColor: 'rgba(99, 102, 241, 0.4)' }}
          >
            <ArrowLeftRight size={15} />
            Compare Providers
          </button>

          <button
            onClick={exportToCSV}
            className="btn-secondary"
            style={{ fontSize: '13px', padding: '8px 16px' }}
          >
            <Download size={15} />
            Export CSV Report
          </button>

          <button
            onClick={onReUpload}
            className="btn-primary"
            style={{ fontSize: '13px', padding: '8px 18px' }}
          >
            <UploadCloud size={16} />
            Upload New Dataset
          </button>
        </div>
      </div>

      {/* 6 Executive KPI Metric Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
        gap: '16px',
        marginBottom: '28px'
      }}>
        {/* Card 1: Total Providers */}
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase' }}>
              Total Providers
            </span>
            <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(99, 102, 241, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Building2 size={16} color="var(--accent-primary)" />
            </div>
          </div>
          <div style={{ fontSize: '28px', fontWeight: '800', color: 'var(--text-primary)' }}>
            {summary?.total_providers?.toLocaleString() || 0}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Audited Across All Claims
          </div>
        </div>

        {/* Card 2: Potential Fraud Flagged (CRITICAL) */}
        <div className="glass-card" style={{ padding: '20px', borderLeft: '4px solid var(--risk-high)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '12px', color: '#fda4af', fontWeight: '600', textTransform: 'uppercase' }}>
              Potential Fraud
            </span>
            <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'var(--risk-high-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <ShieldAlert size={16} color="var(--risk-high)" />
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '28px', fontWeight: '800', color: 'var(--risk-high)' }}>
              {summary?.potential_fraud?.toLocaleString() || 0}
            </span>
            <span className="badge badge-high" style={{ fontSize: '11px', padding: '2px 6px' }}>
              {fraudRate.toFixed(1)}% Flagged
            </span>
          </div>
          <div style={{ fontSize: '12px', color: '#fda4af', marginTop: '4px' }}>
            Exceeds ML Fraud Threshold
          </div>
        </div>

        {/* Card 3: Non-Fraud Providers */}
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase' }}>
              Legitimate Providers
            </span>
            <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'var(--risk-low-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <ShieldCheck size={16} color="var(--risk-low)" />
            </div>
          </div>
          <div style={{ fontSize: '28px', fontWeight: '800', color: 'var(--text-primary)' }}>
            {summary?.non_fraud?.toLocaleString() || 0}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--risk-low)', marginTop: '4px' }}>
            {(100 - fraudRate).toFixed(1)}% Low Risk Rate
          </div>
        </div>

        {/* Card 4: Total Claims */}
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase' }}>
              Total Claims
            </span>
            <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <FileText size={16} color="var(--accent-cyan)" />
            </div>
          </div>
          <div style={{ fontSize: '28px', fontWeight: '800', color: 'var(--text-primary)' }}>
            {summary?.total_claims?.toLocaleString() || 0}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Inpatient & Outpatient
          </div>
        </div>

        {/* Card 5: Total Beneficiaries */}
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase' }}>
              Beneficiaries
            </span>
            <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(139, 92, 246, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Users size={16} color="var(--accent-secondary)" />
            </div>
          </div>
          <div style={{ fontSize: '28px', fontWeight: '800', color: 'var(--text-primary)' }}>
            {summary?.total_beneficiaries?.toLocaleString() || 0}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Patients Covered
          </div>
        </div>

        {/* Card 6: Total Reimbursement */}
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase' }}>
              Total Reimbursement
            </span>
            <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <DollarSign size={16} color="var(--risk-low)" />
            </div>
          </div>
          <div style={{ fontSize: '24px', fontWeight: '800', color: 'var(--text-primary)' }}>
            ${totalReimbursement ? totalReimbursement.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 }) : 0}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Financial Claim Volume
          </div>
        </div>
      </div>

      {/* Row 1: Core ML Visualizations Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
        gap: '20px',
        marginBottom: '28px'
      }}>
        {/* Visualization 1: Risk Tier Distribution */}
        <div className="glass-card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: '700' }}>Risk Tier Breakdown</h3>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Distribution of providers by probability severity</p>
            </div>
            <PieChart size={18} color="var(--accent-cyan)" />
          </div>
          <RiskDistributionChart 
            riskDistribution={analysisData?.risk_distribution} 
            summary={summary} 
            results={results}
          />
        </div>

        {/* Visualization 2: Fraud Classification Comparison */}
        <div className="glass-card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: '700' }}>Fraud vs Non-Fraud Count</h3>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Binary prediction outcome comparison</p>
            </div>
            <BarChart3 size={18} color="var(--accent-primary)" />
          </div>
          <FraudClassificationChart summary={summary} />
        </div>

        {/* Visualization 3: Top Suspicious Providers Score */}
        <div className="glass-card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: '700' }}>Top Suspicious Providers</h3>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Highest fraud probability score ranking</p>
            </div>
            <ShieldAlert size={18} color="var(--risk-high)" />
          </div>
          <TopProvidersBarChart 
            topProviders={analysisData?.top_risk_providers || results} 
            onSelectProvider={onSelectProvider}
          />
        </div>
      </div>

      {/* Row 2: Advanced Geographic & Claim Type Business Analytics */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))',
        gap: '20px',
        marginBottom: '32px'
      }}>
        {/* Geographic Fraud & Reimbursement by State */}
        <div className="glass-card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <MapPin size={18} color="var(--accent-primary)" />
                <h3 style={{ fontSize: '16px', fontWeight: '700' }}>Geographic Fraud Risk Distribution</h3>
              </div>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: '2px 0 0' }}>
                Top states ranked by total reimbursement and high-risk claim volume
              </p>
            </div>
            <span className="badge badge-primary" style={{ fontSize: '11px' }}>Regional Insights</span>
          </div>
          <GeographicStateBarChart 
            geographicInsights={geographic_insights || analysisData?.geographic_insights || []} 
            summary={summary}
          />
        </div>

        {/* Inpatient vs Outpatient Claim Type Comparison */}
        <div className="glass-card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Hospital size={18} color="var(--accent-cyan)" />
                <h3 style={{ fontSize: '16px', fontWeight: '700' }}>Claim Type & Care Setting Analysis</h3>
              </div>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: '2px 0 0' }}>
                Inpatient hospital admissions vs Outpatient ambulatory visits
              </p>
            </div>
            <span className="badge badge-neutral" style={{ fontSize: '11px' }}>
              {claim_type_summary?.inpatient?.count ? `${Math.round(claim_type_summary.inpatient.count / (summary?.total_claims || 1) * 100)}% Inpatient` : 'Care Split'}
            </span>
          </div>
          <ClaimTypeBreakdownChart 
            claimTypeSummary={claim_type_summary || analysisData?.claim_type_summary} 
            summary={summary}
          />
        </div>
      </div>

      {/* Provider Risk Leaderboard & Intelligence Table */}
      <div className="glass-card" style={{ padding: '24px' }}>
        {/* Leaderboard Header & Filters */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '16px',
          marginBottom: '20px'
        }}>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: '800' }}>
              Provider Risk Leaderboard
            </h2>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '2px 0 0' }}>
              Click on any provider row to inspect root-cause analysis, engineered features, and claim anomalies.
            </p>
          </div>

          {/* Search Input */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              background: 'rgba(255, 255, 255, 0.04)',
              border: '1px solid var(--border-color)',
              borderRadius: '10px',
              padding: '8px 14px',
              gap: '8px',
              width: '240px'
            }}>
              <Search size={16} color="var(--text-muted)" />
              <input
                type="text"
                placeholder="Search Provider ID..."
                value={searchTerm}
                onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
                style={{
                  background: 'transparent',
                  border: 'none',
                  outline: 'none',
                  color: 'var(--text-primary)',
                  fontSize: '13px',
                  width: '100%'
                }}
              />
            </div>

            {/* Filter Buttons */}
            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
              {[
                { id: 'all', label: 'All' },
                { id: 'fraud', label: '🚨 Potential Fraud' },
                { id: 'high_risk', label: 'High Risk (≥75%)' },
                { id: 'medium_risk', label: 'Medium (40-74%)' },
                { id: 'low_risk', label: 'Low (<40%)' }
              ].map(f => (
                <button
                  key={f.id}
                  type="button"
                  onClick={() => { setFilterType(f.id); setCurrentPage(1); }}
                  className="btn-secondary"
                  style={{
                    padding: '6px 12px',
                    fontSize: '12px',
                    borderRadius: '8px',
                    background: filterType === f.id ? 'var(--accent-primary)' : 'rgba(255, 255, 255, 0.04)',
                    borderColor: filterType === f.id ? 'var(--accent-primary)' : 'var(--border-color)',
                    color: filterType === f.id ? '#fff' : 'var(--text-secondary)'
                  }}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Providers Table */}
        <div style={{ overflowX: 'auto', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
          <table className="custom-table">
            <thead>
              <tr>
                <th style={{ cursor: 'pointer' }} onClick={() => handleSort('Provider')}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    Provider ID
                    <ArrowUpDown size={13} />
                  </div>
                </th>
                <th>Prediction Status</th>
                <th style={{ cursor: 'pointer' }} onClick={() => handleSort('FraudProbability')}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    Fraud Probability
                    <ArrowUpDown size={13} />
                  </div>
                </th>
                <th>Risk Meter</th>
                <th>Risk Level</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {paginatedProviders.length > 0 ? (
                paginatedProviders.map((p, idx) => {
                  const prob = Math.round(p.FraudProbability ?? 0);
                  const isFraud = p.Prediction === 'Potential Fraud';
                  const isHigh = prob >= 75;
                  const isMedium = prob >= 40 && prob < 75;
                  const riskColor = isHigh ? 'var(--risk-high)' : isMedium ? 'var(--risk-medium)' : 'var(--risk-low)';

                  return (
                    <tr 
                      key={idx} 
                      onClick={() => onSelectProvider(p.Provider)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td style={{ fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--accent-cyan)' }}>
                        {p.Provider}
                      </td>
                      <td>
                        <span className={`badge ${isFraud ? 'badge-high' : 'badge-low'}`}>
                          {isFraud ? <AlertTriangle size={12} /> : <ShieldCheck size={12} />}
                          {p.Prediction}
                        </span>
                      </td>
                      <td>
                        <span style={{ fontWeight: '700', fontSize: '15px', color: riskColor }}>
                          {prob}%
                        </span>
                      </td>
                      <td style={{ width: '160px' }}>
                        <div className="risk-meter">
                          <div 
                            className="risk-meter-fill" 
                            style={{ width: `${prob}%`, backgroundColor: riskColor }} 
                          />
                        </div>
                      </td>
                      <td>
                        <span className={`badge ${isHigh ? 'badge-high' : isMedium ? 'badge-medium' : 'badge-low'}`}>
                          {isHigh ? 'High Risk' : isMedium ? 'Medium Risk' : 'Low Risk'}
                        </span>
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectProvider(p.Provider);
                          }}
                          className="btn-secondary"
                          style={{ padding: '6px 12px', fontSize: '12px', borderRadius: '6px' }}
                        >
                          <span>Inspect</span>
                          <ExternalLink size={13} />
                        </button>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                    No providers matched your filter criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Toolbar */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginTop: '16px',
          fontSize: '13px',
          color: 'var(--text-muted)'
        }}>
          <div>
            Showing {(currentPage - 1) * itemsPerPage + 1} to {Math.min(currentPage * itemsPerPage, filteredProviders.length)} of {filteredProviders.length} providers
          </div>

          <div style={{ display: 'flex', gap: '6px' }}>
            <button
              onClick={() => setCurrentPage(p => Math.max(p - 1, 1))}
              disabled={currentPage === 1}
              className="btn-secondary"
              style={{ padding: '6px 12px', fontSize: '12px' }}
            >
              Previous
            </button>
            <div style={{ display: 'flex', alignItems: 'center', padding: '0 10px', fontWeight: '600', color: 'var(--text-primary)' }}>
              {currentPage} / {totalPages}
            </div>
            <button
              onClick={() => setCurrentPage(p => Math.min(p + 1, totalPages))}
              disabled={currentPage === totalPages}
              className="btn-secondary"
              style={{ padding: '6px 12px', fontSize: '12px' }}
            >
              Next
            </button>
          </div>
        </div>

      </div>

      {/* Side-by-Side Provider Comparison Modal */}
      {showCompareModal && (
        <ProviderCompareModal
          initialProviderId={compareInitialProvider}
          allProviders={results || []}
          analysisId={analysis_id}
          backendUrl={import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:8000"}
          onClose={() => setShowCompareModal(false)}
          onSelectProvider={(id) => onSelectProvider(id)}
        />
      )}

    </div>
  );
}
