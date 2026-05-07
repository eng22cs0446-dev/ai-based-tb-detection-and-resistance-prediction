export default function Card({ title, children, style = {} }) {
  return (
    <div style={{
      ...style,
      backgroundColor: 'var(--color-bg-surface)',
      borderRadius: 'var(--radius-lg)',
      padding: '1.5rem',
      border: '1px solid var(--color-border)',
      boxShadow: 'var(--shadow-sm)',
    }}>
      {title && <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: 'var(--color-text-secondary)' }}>{title}</h3>}
      {children}
    </div>
  );
}
