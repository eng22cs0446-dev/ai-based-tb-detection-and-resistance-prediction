import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../api/api';
import { useApi } from '../hooks/useApi';
import Card from '../components/Card';
import Badge from '../components/Badge';
import { Download, ChevronLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

function PatientHistoryList({ patientId, currentCaseId }) {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!patientId) return;
    api.getPatientCases(patientId)
      .then(data => setCases(data.cases || []))
      .catch(err => console.error("Error fetching patient history:", err))
      .finally(() => setLoading(false));
  }, [patientId]);

  if (loading) return <div style={{ fontSize: '0.9rem', color: 'var(--color-text-secondary)' }}>Loading history...</div>;
  if (!cases.length || (cases.length === 1 && cases[0].case_id === currentCaseId)) {
    return <div style={{ fontSize: '0.9rem', color: 'var(--color-text-secondary)' }}>No other scans found for this patient.</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
      {cases.filter(c => c.case_id !== currentCaseId).map(c => (
        <a key={c.case_id} href={`/cases/${c.case_id}`} style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '0.75rem', backgroundColor: 'var(--color-bg-elevated)',
          borderRadius: 'var(--radius-md)', textDecoration: 'none', border: '1px solid var(--color-border)'
        }}>
          <div>
            <div style={{ color: 'var(--color-text-primary)', fontSize: '0.9rem', fontWeight: 500 }}>
              {new Date(c.timestamp).toLocaleDateString()}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>ID: {c.case_id.slice(0, 8)}</div>
          </div>
          <Badge 
            text={c.tb_probability > 0.5 ? 'Detected' : 'Normal'} 
            color={c.tb_probability > 0.5 ? 'var(--color-danger)' : 'var(--color-success)'} 
            bg="transparent" 
          />
        </a>
      ))}
    </div>
  );
}

export default function CaseDetails() {
  const { caseId } = useParams();
  const { execute: fetchDetails, data, loading } = useApi(api.getCaseDetails);
  const [activeLayer, setActiveLayer] = useState('heatmap'); // original, heatmap, overlay

  useEffect(() => {
    fetchDetails(caseId);
  }, [caseId]);

  if (loading || !data) {
    return <div>Loading case details...</div>;
  }

  const {
    patient,
    tb_detection: tb,
    drug_resistance: dr,
    clinical_risk: risk,
    timestamp
  } = data;

  const isDetected = tb?.probability > 0.5;
  const tbPercent = Math.round((tb?.probability || 0) * 100);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <Link to="/dashboard" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-text-secondary)', marginBottom: '0.5rem', textDecoration: 'none' }}>
            <ChevronLeft size={16} /> Back
          </Link>
          <h2 style={{ fontSize: '2rem', margin: 0, display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            Case Details
          </h2>
          <p style={{ color: 'var(--color-text-muted)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
            ID: {caseId} • Analyzed on {new Date(timestamp).toLocaleString()}
          </p>
        </div>
        
        <a href={api.getReportDownloadUrl(caseId)} target="_blank" rel="noreferrer" style={{
          display: 'flex', alignItems: 'center', gap: '0.5rem', textDecoration: 'none',
          padding: '0.75rem 1rem', backgroundColor: 'var(--color-bg-base)', border: '1px solid var(--color-border)',
          color: 'var(--color-text-primary)', borderRadius: 'var(--radius-md)', fontWeight: 500
        }}>
          <Download size={18} /> Download Report
        </a>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem' }}>
        
        {/* Left Column: XRAY Viewer */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <Card style={{ padding: '0', overflow: 'hidden' }}>
            <div style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--color-border)' }}>
              <h3 style={{ margin: 0, fontSize: '1.1rem' }}>Chest Radiograph (X-ray)</h3>
              <div style={{ display: 'flex', gap: '0.5rem', backgroundColor: 'var(--color-bg-elevated)', padding: '0.25rem', borderRadius: 'var(--radius-md)' }}>
                {['original', 'heatmap', 'overlay'].map(mode => (
                  <button key={mode} onClick={() => setActiveLayer(mode)} style={{
                    padding: '0.35rem 0.75rem',
                    textTransform: 'capitalize', fontSize: '0.85rem',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: activeLayer === mode ? 'var(--color-bg-surface)' : 'transparent',
                    color: activeLayer === mode ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                  }}>
                    {mode}
                  </button>
                ))}
              </div>
            </div>
            
            <div style={{ height: '400px', backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
              {/* Dummy XRAY Area */}
              <div style={{ 
                width: '100%', height: '100%', 
                background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(0,0,0,1) 100%)',
                opacity: activeLayer === 'original' ? 1 : 0.6
              }} />
              
              {(activeLayer === 'heatmap' || activeLayer === 'overlay') && isDetected && (
                <div style={{
                  position: 'absolute', top: '30%', left: '30%',
                  width: '150px', height: '150px',
                  background: 'radial-gradient(circle, rgba(245,101,101,0.8) 0%, rgba(245,101,101,0) 70%)',
                  borderRadius: '50%', filter: 'blur(20px)', zIndex: 10
                }} />
              )}
              
              <div style={{ position: 'absolute', bottom: '1rem', left: '1rem' }}>
                <span style={{ backgroundColor: 'rgba(21,26,34,0.8)', padding: '0.25rem 0.75rem', borderRadius: 'var(--radius-xl)', fontSize: '0.75rem', fontWeight: 600 }}>
                  AI ANALYSIS SUPPORT LAYER
                </span>
              </div>
            </div>
          </Card>

          <Card title="AI Interpretation">
            <div style={{ padding: '1rem', backgroundColor: 'var(--color-bg-base)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}>
              <p style={{ margin: 0, color: 'var(--color-text-primary)' }}>{data.finding_text}</p>
            </div>
          </Card>
        </div>

        {/* Right Column: Stats & Information */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          
          <Card>
            <h3 style={{ fontSize: '0.9rem', color: 'var(--color-text-secondary)', marginBottom: '1rem' }}>TB detection result</h3>
            <div style={{ fontSize: '3.5rem', fontWeight: 'bold', lineHeight: 1, marginBottom: '0.5rem' }}>
              {tbPercent}<span style={{ fontSize: '1.5rem', color: 'var(--color-text-muted)' }}>%</span>
            </div>
            <div style={{ marginBottom: '1rem' }}>
              <Badge 
                text={isDetected ? 'Detected' : 'Not Detected'} 
                color={isDetected ? 'var(--color-danger)' : 'var(--color-success)'} 
                bg={isDetected ? 'rgba(245,101,101,0.1)' : 'rgba(72,187,120,0.1)'} 
              />
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
              Threshold used: {tb?.threshold_used} ({tb?.calibrated ? 'Calibrated' : 'Uncalibrated'})
            </div>
          </Card>

          <Card>
            <h3 style={{ fontSize: '0.9rem', color: 'var(--color-text-secondary)', marginBottom: '0.5rem' }}>Drug resistance prediction</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-success)', fontWeight: 600, fontSize: '1.1rem', marginBottom: '0.5rem' }}>
              <span>{dr?.prediction}</span>
            </div>
            {dr?.is_demo_mode && (
              <div style={{ fontSize: '0.8rem', color: 'var(--color-warning)' }}>
                ⚠️ Predicted based on patient demographics (Demo Mode)
              </div>
            )}
            
            <div style={{ marginTop: '1.5rem' }}>
              <h3 style={{ fontSize: '0.9rem', color: 'var(--color-text-secondary)', marginBottom: '0.5rem' }}>Clinical risk band</h3>
              <div style={{ color: 'var(--color-accent)', fontWeight: 500 }}>
                {risk?.band} (Score: {risk?.score})
              </div>
            </div>
          </Card>

          <Card title="Patient Details">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>Patient Id</div>
                <div style={{ fontWeight: 500 }}>{patient?.patient_id || 'Unknown'}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>Gender</div>
                <div style={{ fontWeight: 500 }}>{patient?.gender || 'Unknown'}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>Age</div>
                <div style={{ fontWeight: 500 }}>{patient?.age ? `${patient.age} years` : 'Unknown'}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>District</div>
                <div style={{ fontWeight: 500 }}>{patient?.district || 'Unknown'}</div>
              </div>
            </div>
          </Card>
          
          <Card title="Patient History">
            <h3 style={{ fontSize: '0.9rem', color: 'var(--color-text-secondary)', margin: 0 }}>Previous scans</h3>
            <PatientHistoryList patientId={patient?.patient_id} currentCaseId={caseId} />
          </Card>

        </div>
      </div>
    </div>
  );
}
