import { useState, useEffect, useCallback } from 'react';
import { healthCheck } from '../lib/api';

/**
 * Polls backend health endpoint every 15s.
 * Returns status: 'checking' | 'online' | 'offline'
 */
export function useBackendStatus(intervalMs = 15000) {
  const [status, setStatus] = useState('checking');
  const [lastChecked, setLastChecked] = useState(null);

  const check = useCallback(async () => {
    try {
      await healthCheck();
      setStatus('online');
    } catch {
      setStatus('offline');
    } finally {
      setLastChecked(new Date());
    }
  }, []);

  useEffect(() => {
    check();
    const timer = setInterval(check, intervalMs);
    return () => clearInterval(timer);
  }, [check, intervalMs]);

  return { status, lastChecked, recheck: check };
}
