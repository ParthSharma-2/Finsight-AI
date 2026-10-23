import { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Nav from './components/Nav';
import Footer from './components/Footer';
import CopilotDrawer from './components/CopilotDrawer';
import Home from './pages/Home';
import Terminal from './pages/Terminal';
import Markets from './pages/Markets';
import Research from './pages/Research';
import NotFound from './pages/NotFound';

function AppLayout() {
  const [copilotOpen, setCopilotOpen] = useState(false);

  return (
    <div className="flex flex-col min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-50 transition-colors duration-300">
      <Nav onToggleCopilot={() => setCopilotOpen(prev => !prev)} />
      
      <main className="flex-1 pt-14">
        <Routes>
          <Route path="/" element={<Home onOpenCopilot={() => setCopilotOpen(true)} />} />
          <Route path="/terminal" element={<Terminal />} />
          <Route path="/markets" element={<Markets onOpenCopilot={() => setCopilotOpen(true)} />} />
          <Route path="/research" element={<Research onOpenCopilot={() => setCopilotOpen(true)} />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>

      <Footer />

      {/* Slide-out Persistent Copilot Drawer */}
      <CopilotDrawer isOpen={copilotOpen} onClose={() => setCopilotOpen(false)} />
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <AppLayout />
      </BrowserRouter>
    </ThemeProvider>
  );
}
