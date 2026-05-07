import { useEffect } from 'react';
import { api } from '../api/api';
import { useApi } from '../hooks/useApi';
import Card from '../components/Card';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

export default function AnalyticsView() {
  const { execute: fetchStats, data: stats, loading } = useApi(api.getDashboardStats);

  useEffect(() => {
    // Fetch stats for the whole year 
    fetchStats(365);
  }, []);

  if (loading || !stats) {
    return <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--color-text-secondary)' }}>Loading analytics...</div>;
  }

  // ── District Breakdown Chart ──
  const districtLabels = stats.by_district?.map(d => d.district) || [];
  const districtTotal = stats.by_district?.map(d => d.total) || [];
  const districtDetected = stats.by_district?.map(d => d.tb_detected) || [];

  const districtChartData = {
    labels: districtLabels,
    datasets: [
      {
        label: 'TB Detected',
        data: districtDetected,
        backgroundColor: 'var(--color-danger)',
        borderRadius: 4,
      },
      {
        label: 'Total Scans',
        data: districtTotal,
        backgroundColor: 'var(--color-bg-surface)',
        borderColor: 'var(--color-border)',
        borderWidth: 1,
        borderRadius: 4,
      }
    ],
  };

  const barOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { 
        position: 'top', 
        labels: { color: 'var(--color-text-secondary)' } 
      },
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
        ticks: { color: 'var(--color-text-secondary)' }
      }
    }
  };

  // ── Drug Resistance Pie Chart ──
  const drKeys = Object.keys(stats.dr_breakdown || {});
  const drValues = Object.values(stats.dr_breakdown || {});

  // Generate some pretty colors based on risk type
  const getDrColor = (label) => {
    if (label.includes('XDR')) return 'rgba(245, 101, 101, 0.9)'; // Red
    if (label.includes('MDR')) return 'rgba(237, 137, 54, 0.9)'; // Orange
    if (label.includes('INH')) return 'rgba(236, 201, 75, 0.9)'; // Yellow
    if (label.includes('RIF')) return 'rgba(159, 122, 234, 0.9)'; // Purple
    return 'rgba(72, 187, 120, 0.9)'; // Green
  };

  const drChartData = {
    labels: drKeys,
    datasets: [
      {
        data: drValues,
        backgroundColor: drKeys.map(getDrColor),
        borderColor: 'var(--color-bg-base)',
        borderWidth: 2,
      }
    ],
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: '65%',
    plugins: {
      legend: {
        position: 'right',
        labels: { color: 'var(--color-text-secondary)', padding: 20 }
      },
      tooltip: {
        backgroundColor: 'var(--color-bg-base)',
        titleColor: 'var(--color-text-primary)',
        bodyColor: 'var(--color-text-secondary)',
        borderColor: 'var(--color-border)',
        borderWidth: 1,
      }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '2rem', margin: 0 }}>Analytics</h2>
          <p style={{ color: 'var(--color-text-muted)', margin: '0.25rem 0 0 0' }}>Data insights covering the last 365 days</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem' }}>
        <Card title="Total Screened">
          <div style={{ fontSize: '2.5rem', fontWeight: 600 }}>{stats.total_cases}</div>
        </Card>
        <Card title="TB Positive">
          <div style={{ fontSize: '2.5rem', fontWeight: 600, color: 'var(--color-danger)' }}>{stats.tb_detected}</div>
        </Card>
        <Card title="Positivity Rate">
          <div style={{ fontSize: '2.5rem', fontWeight: 600 }}>
            {((stats.detection_rate || 0) * 100).toFixed(1)}%
          </div>
        </Card>
        <Card title="Districts Reached">
          <div style={{ fontSize: '2.5rem', fontWeight: 600 }}>
            {stats.by_district?.length || 0}
          </div>
        </Card>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem' }}>
        
        <Card title="Geographical Heatmap (District Wise)">
          <div style={{ height: '350px', marginTop: '1rem' }}>
            <Bar options={barOptions} data={districtChartData} />
          </div>
        </Card>

        <Card title="Drug Resistance Breakdown">
          <div style={{ height: '350px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            {drKeys.length > 0 ? (
              <Doughnut options={doughnutOptions} data={drChartData} />
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--color-text-muted)' }}>No DR data available</div>
            )}
            {drKeys.length > 0 && (
              <div style={{ textAlign: 'center', marginTop: '1rem', fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
                Shows predicted resistance across all detected cases.
              </div>
            )}
          </div>
        </Card>
        
      </div>
    </div>
  );
}
