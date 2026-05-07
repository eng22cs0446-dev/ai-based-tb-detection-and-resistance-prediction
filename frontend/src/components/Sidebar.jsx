import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, PlusCircle, History, BarChart3, User } from 'lucide-react';

export default function Sidebar() {
  const location = useLocation();

  const navItems = [
    { path: '/dashboard', label: 'Dashboard', icon: <LayoutDashboard size={20} /> },
    { path: '/new-case', label: 'New Case', icon: <PlusCircle size={20} /> },
    { path: '/history', label: 'History', icon: <History size={20} /> },
    { path: '/analytics', label: 'Analytics', icon: <BarChart3 size={20} /> },
  ];

  return (
    <aside style={{
      width: 'var(--sidebar-width)',
      height: '100vh',
      position: 'fixed',
      left: 0,
      top: 0,
      backgroundColor: 'var(--color-bg-base)',
      borderRight: '1px solid var(--color-border)',
      display: 'flex',
      flexDirection: 'column',
      padding: '1.5rem 1rem'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '3rem', padding: '0 0.5rem' }}>
        <div style={{
          width: 32, height: 32, 
          backgroundColor: 'var(--color-bg-elevated)',
          borderRadius: 8,
          display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
          {/* Logo icon placeholder */}
          <div style={{ width: 16, height: 16, border: '2px solid white', borderRadius: 4 }}></div>
        </div>
        <h1 style={{ fontSize: '1.25rem', margin: 0 }}>NexTB</h1>
      </div>

      <nav style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {navItems.map(item => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              style={{
                display: 'flex', alignItems: 'center', gap: '0.75rem',
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                color: isActive ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                backgroundColor: isActive ? 'var(--color-bg-surface)' : 'transparent',
                fontWeight: isActive ? 500 : 400,
                textDecoration: 'none',
                transition: 'all 0.2s'
              }}
            >
              {item.icon}
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div style={{ marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--color-border)' }}>
        <Link to="/account" style={{
          display: 'flex', alignItems: 'center', gap: '0.75rem',
          padding: '0.75rem 0.5rem',
          color: 'var(--color-text-secondary)',
          textDecoration: 'none'
        }}>
          <User size={20} />
          My Account
        </Link>
      </div>
    </aside>
  );
}
