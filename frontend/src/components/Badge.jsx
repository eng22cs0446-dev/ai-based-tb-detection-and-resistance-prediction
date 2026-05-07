export default function Badge({ text, color = 'var(--color-text-secondary)', bg = 'var(--color-bg-base)' }) {
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      padding: '0.25rem 0.75rem',
      borderRadius: 'var(--radius-xl)',
      fontSize: '0.85rem',
      fontWeight: 500,
      color: color,
      backgroundColor: bg,
      border: `1px solid ${color}40`, // 25% opacity border
    }}>
      {text}
    </span>
  );
}
