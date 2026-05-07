import { useEffect } from 'react';
import { api } from '../api/api';
import { useApi } from '../hooks/useApi';
import Card from '../components/Card';
import Badge from '../components/Badge';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { ArrowUpRight } from 'lucide-react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export default function Dashboard() {
  const { execute: fetchStats, data: stats, loading: statsLoading } = useApi(api.getDashboardStats);
  const { execute: fetchCases, data: casesData, loading: casesLoading } = useApi(api.getCases);

  useEffect(() => {
    fetchStats(30);
    fetchCases(1, 10); // recent 10
  }, []);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'var(--color-bg-base)',
        titleColor: 'var(--color-text-primary)',
        bodyColor: 'var(--color-text-secondary)',
        borderColor: 'var(--color-border)',
        borderWidth: 1,
      }
    },
    scales: {
      x: { 
        grid: { display: false, drawBorder: false },
        ticks: { color: 'var(--color-text-secondary)' }
      },
      y: { 
        grid: { color: 'var(--color-border)', drawBorder: false },
        ticks: { color: 'var(--color-text-secondary)', stepSize: 1 }
      }
    }
  };

  const chartData = {
    labels: stats?.by_week.map(w => w.week) || [],
    datasets: [
      {
        label: 'Total Cases',
        data: stats?.by_week.map(w => w.total) || [],
        backgroundColor: 'var(--color-text-secondary)',
        borderRadius: 4,
      },
      {
        label: 'TB Detected',
        data: stats?.by_week.map(w => w.tb_detected) || [],
        backgroundColor: 'var(--color-danger)',
        borderRadius: 4,
      }
    ],
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '2rem', margin: 0 }}>Dashboard</h2>
        </div>
        <div>
          <button style={{
            display: 'flex', alignItems: 'center', gap: '0.5rem',
            padding: '0.5rem 1rem',
            backgroundColor: 'var(--color-bg-base)',
            border: '1px solid var(--color-border)',
            color: 'var(--color-text-primary)',
            borderRadius: 'var(--radius-md)'
          }}>
            Download
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
        <Card title="Total Cases">
          <div style={{ fontSize: '2.5rem', fontWeight: 600 }}>{statsLoading ? '-' : stats?.total_cases || 0}</div>
        </Card>
        <Card title="TB Positive">
          <div style={{ fontSize: '2.5rem', fontWeight: 600 }}>{statsLoading ? '-' : stats?.tb_detected || 0}</div>
        </Card>
        <Card title="Avg. Confidence">
          <div style={{ fontSize: '2.5rem', fontWeight: 600 }}>
            {statsLoading ? '-' : `${((stats?.avg_probability || 0) * 100).toFixed(1)}%`}
          </div>
        </Card>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem' }}>
        <Card title="Case Overview">
          <div style={{ height: '300px' }}>
            <Bar options={chartOptions} data={chartData} />
          </div>
        </Card>

        <Card title="Recent Patients">
          {casesLoading ? <p>Loading...</p> : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 24px',
                color: 'var(--color-text-secondary)', fontSize: '0.85rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--color-border)'
              }}>
                <div>Patient</div>
                <div>Result</div>
                <div>Date</div>
                <div></div>
              </div>
              {casesData?.cases?.map(c => (
                <div key={c.case_id} style={{
                  display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 24px', alignItems: 'center', fontSize: '0.9rem'
                }}>
                  <div>
                    <div style={{ fontWeight: 500, color: 'var(--color-text-primary)' }}>{c.patient_id}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>{c.case_id.slice(0, 8)}</div>
                  </div>
                  <div>
                    <Badge 
                      text={c.tb_probability > 0.5 ? 'Detected' : 'Not Detected'} 
                      color={c.tb_probability > 0.5 ? 'var(--color-danger)' : 'var(--color-success)'} 
                      bg="transparent" 
                    />
                  </div>
                  <div style={{ color: 'var(--color-text-secondary)' }}>
                    {new Date(c.timestamp).toLocaleDateString()}
                  </div>
                  <a href={`/cases/${c.case_id}`} style={{ color: 'var(--color-text-secondary)', display: 'flex' }}>
                    <ArrowUpRight size={16} />
                  </a>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
