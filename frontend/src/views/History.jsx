import { useEffect, useState } from 'react';
import { api } from '../api/api';
import Card from '../components/Card';
import Badge from '../components/Badge';
import { Trash2, Eye, Search, AlertCircle } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function HistoryView() {
  const [cases, setCases] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const limit = 15;

  const fetchCases = async () => {
    setLoading(true);
    try {
      const data = await api.getCases(page, limit);
      setCases(data.cases || []);
      setTotal(data.total || 0);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to fetch cases');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCases();
  }, [page]);

  const handleDelete = async (caseId) => {
    if (!window.confirm(`Are you sure you want to delete case ${caseId}? This cannot be undone.`)) {
      return;
    }
    
    try {
      await api.deleteCase(caseId);
      // Refresh the current page
      fetchCases();
    } catch (err) {
      alert(`Error deleting case: ${err.message}`);
    }
  };

  const totalPages = Math.ceil(total / limit);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '2rem', margin: 0 }}>Case History</h2>
          <p style={{ color: 'var(--color-text-muted)', margin: '0.25rem 0 0 0' }}>Manage and review all patient scans</p>
        </div>
      </div>

      <Card style={{ padding: '0', overflow: 'hidden' }}>
        <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--color-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ position: 'relative', width: '300px' }}>
            <Search size={16} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--color-text-muted)' }} />
            <input 
              type="text" 
              placeholder="Search by Patient ID (Coming Soon...)" 
              disabled
              style={{
                width: '100%', padding: '0.5rem 0.5rem 0.5rem 2rem',
                backgroundColor: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)',
                borderRadius: 'var(--radius-md)', color: 'var(--color-text-primary)'
              }}
            />
          </div>
          <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
            Showing {cases.length > 0 ? (page - 1) * limit + 1 : 0} - {Math.min(page * limit, total)} of {total} cases
          </div>
        </div>

        {error && (
          <div style={{ padding: '2rem', display: 'flex', justifyContent: 'center', color: 'var(--color-danger)' }}>
            <AlertCircle size={24} style={{ marginRight: '0.5rem' }} /> {error}
          </div>
        )}

        {loading ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--color-text-secondary)' }}>Loading cases...</div>
        ) : cases.length === 0 ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--color-text-secondary)' }}>No cases found.</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ backgroundColor: 'var(--color-bg-surface)', color: 'var(--color-text-secondary)', fontSize: '0.85rem' }}>
                  <th style={{ padding: '1rem', borderBottom: '1px solid var(--color-border)', fontWeight: 500 }}>Case ID</th>
                  <th style={{ padding: '1rem', borderBottom: '1px solid var(--color-border)', fontWeight: 500 }}>Patient ID</th>
                  <th style={{ padding: '1rem', borderBottom: '1px solid var(--color-border)', fontWeight: 500 }}>District</th>
                  <th style={{ padding: '1rem', borderBottom: '1px solid var(--color-border)', fontWeight: 500 }}>Result</th>
                  <th style={{ padding: '1rem', borderBottom: '1px solid var(--color-border)', fontWeight: 500 }}>Risk Band</th>
                  <th style={{ padding: '1rem', borderBottom: '1px solid var(--color-border)', fontWeight: 500 }}>Date</th>
                  <th style={{ padding: '1rem', borderBottom: '1px solid var(--color-border)', fontWeight: 500, textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {cases.map((c) => {
                  const isDetected = c.tb_probability > 0.5;
                  
                  let riskColor = 'var(--color-text-muted)';
                  let riskBg = 'transparent';
                  if (c.risk_band === 'HIGH') { riskColor = 'var(--color-danger)'; riskBg = 'rgba(245,101,101,0.1)'; }
                  else if (c.risk_band === 'MODERATE') { riskColor = 'var(--color-warning)'; riskBg = 'rgba(227,179,65,0.1)'; }
                  else { riskColor = 'var(--color-success)'; riskBg = 'rgba(63,185,80,0.1)'; }

                  return (
                    <tr key={c.case_id} style={{ borderBottom: '1px solid var(--color-border)', transition: 'background-color 0.2s', ':hover': { backgroundColor: 'var(--color-bg-surface)' }}}>
                      <td style={{ padding: '1rem', fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>
                        {c.case_id.slice(0, 8)}
                      </td>
                      <td style={{ padding: '1rem', fontWeight: 500 }}>{c.patient_id}</td>
                      <td style={{ padding: '1rem', color: 'var(--color-text-secondary)' }}>{c.district || 'Unknown'}</td>
                      <td style={{ padding: '1rem' }}>
                        <Badge 
                          text={isDetected ? 'Detected' : 'Not Detected'} 
                          color={isDetected ? 'var(--color-danger)' : 'var(--color-success)'} 
                          bg="transparent" 
                        />
                      </td>
                      <td style={{ padding: '1rem' }}>
                        <Badge text={c.risk_band} color={riskColor} bg={riskBg} />
                      </td>
                      <td style={{ padding: '1rem', color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
                        {new Date(c.timestamp).toLocaleDateString()}
                      </td>
                      <td style={{ padding: '1rem', textAlign: 'right' }}>
                        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
                          <Link to={`/cases/${c.case_id}`} style={{
                            padding: '0.4rem', color: 'var(--color-text-secondary)', 
                            backgroundColor: 'var(--color-bg-surface)', borderRadius: '4px',
                            display: 'flex', alignItems: 'center', justifyContent: 'center'
                          }} title="View Case">
                            <Eye size={16} />
                          </Link>
                          <button 
                            onClick={() => handleDelete(c.case_id)}
                            style={{
                              padding: '0.4rem', color: 'var(--color-danger)', 
                              backgroundColor: 'rgba(245,101,101,0.1)', border: 'none', borderRadius: '4px',
                              display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer'
                            }} 
                            title="Delete Case"
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
        
        {/* Pagination */}
        {!loading && totalPages > 1 && (
          <div style={{ padding: '1.5rem', borderTop: '1px solid var(--color-border)', display: 'flex', justifyContent: 'center', gap: '0.5rem' }}>
            <button 
              disabled={page === 1}
              onClick={() => setPage(p => p - 1)}
              style={{ 
                padding: '0.5rem 1rem', borderRadius: 'var(--radius-md)', 
                backgroundColor: page === 1 ? 'transparent' : 'var(--color-bg-surface)',
                color: page === 1 ? 'var(--color-text-muted)' : 'var(--color-text-primary)',
                border: '1px solid var(--color-border)', cursor: page === 1 ? 'not-allowed' : 'pointer'
              }}
            >
              Previous
            </button>
            <span style={{ display: 'flex', alignItems: 'center', padding: '0 1rem', color: 'var(--color-text-secondary)' }}>
              Page {page} of {totalPages}
            </span>
            <button 
              disabled={page === totalPages}
              onClick={() => setPage(p => p + 1)}
              style={{ 
                padding: '0.5rem 1rem', borderRadius: 'var(--radius-md)', 
                backgroundColor: page === totalPages ? 'transparent' : 'var(--color-bg-surface)',
                color: page === totalPages ? 'var(--color-text-muted)' : 'var(--color-text-primary)',
                border: '1px solid var(--color-border)', cursor: page === totalPages ? 'not-allowed' : 'pointer'
              }}
            >
              Next
            </button>
          </div>
        )}
      </Card>
      
    </div>
  );
}
