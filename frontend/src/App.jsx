import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './views/Dashboard';
import NewCase from './views/NewCase';
import CaseDetails from './views/CaseDetails';
import HistoryView from './views/History';
import AnalyticsView from './views/Analytics';
import './styles/global.css';

function App() {
  return (
    <Router>
      <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: 'var(--color-bg-base)' }}>
        <Sidebar />
        <main id="app-content" style={{ flex: 1, padding: '2rem', marginLeft: 'var(--sidebar-width)' }}>
          <div style={{ maxWidth: 'var(--page-max-width)', margin: '0 auto' }}>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/new-case" element={<NewCase />} />
              <Route path="/cases/:caseId" element={<CaseDetails />} />
              <Route path="/history" element={<HistoryView />} />
              <Route path="/analytics" element={<AnalyticsView />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;
