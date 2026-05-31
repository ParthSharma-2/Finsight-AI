import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 pt-12 grid-bg">
      <div className="text-center">
        <div className="mono text-8xl font-700 text-border mb-4">404</div>
        <div className="mono text-xl text-dim mb-2">TERMINAL: PATH NOT FOUND</div>
        <div className="mono text-sm text-muted mb-8">
          The requested route does not exist in this system.
        </div>
        <Link to="/">
          <button className="btn-primary">← RETURN TO BASE</button>
        </Link>
      </div>
    </div>
  );
}
