import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import Nav from './components/Nav';
import TickerTape from './components/TickerTape';
import Footer from './components/Footer';
import Home from './pages/Home';
import Terminal from './pages/Terminal';
import Markets from './pages/Markets';
import Research from './pages/Research';
import NotFound from './pages/NotFound';

// CRT overlay for terminal aesthetic
function CRTOverlay() {
  return <div className="crt-overlay animate-flicker pointer-events-none" />;
}

function AppLayout() {
  const location = useLocation();
  const isTerminal = location.pathname === '/terminal';

  return (
    <div className="flex flex-col min-h-screen bg-terminal-bg">
      <CRTOverlay />
      <Nav />
      <TickerTape />

      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/terminal" element={<Terminal />} />
          <Route path="/markets" element={<Markets />} />
          <Route path="/research" element={<Research />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>

      {!isTerminal && <Footer />}
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}
