// ============================================================
// FinSight AI — API Client
// All backend communication is centralized here.
// Base URL reads from env var; falls back to localhost for dev.
// ============================================================

const BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

/**
 * Core fetch wrapper with error handling and JSON parsing.
 */
async function apiFetch(path, options = {}) {
  const url = `${BASE_URL}${path}`;

  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  const response = await fetch(url, defaultOptions);

  if (!response.ok) {
    let errorMsg = `HTTP ${response.status}`;
    try {
      const err = await response.json();
      errorMsg = err.detail || err.message || errorMsg;
    } catch {
      // fallback to status text
      errorMsg = response.statusText || errorMsg;
    }
    throw new Error(errorMsg);
  }

  return response.json();
}

// ============================================================
// Health Check
// ============================================================

/**
 * Ping the backend to verify connectivity.
 * GET /
 * Returns: { status: "ok", ... }
 */
export async function healthCheck() {
  return apiFetch('/');
}

// ============================================================
// Chat / AI Interaction
// ============================================================

/**
 * Send a message to the AI assistant.
 * POST /chat
 * Body: { message: string, session_id?: string, context?: object }
 * Returns: { response: string, session_id?: string, sources?: array }
 */
export async function sendMessage(message) {

  return apiFetch('/chat', {

    method: 'POST',

    body: JSON.stringify({
      query: message,
    }),

  });
}

// ============================================================
// Market Data (future endpoints)
// ============================================================

/**
 * Fetch stock quote data.
 * GET /market/quote?symbol=AAPL
 * Returns: { symbol, price, change, change_pct, volume, ... }
 */
export async function getStockQuote(symbol) {
  return apiFetch(`/market/quote?symbol=${encodeURIComponent(symbol)}`);
}

/**
 * Fetch stock chart data.
 * GET /market/chart?symbol=AAPL&period=1mo
 */
export async function getStockChart(symbol, period = '1mo') {
  return apiFetch(`/market/chart?symbol=${encodeURIComponent(symbol)}&period=${period}`);
}

/**
 * Search for securities.
 * GET /market/search?q=apple
 */
export async function searchSecurities(query) {
  return apiFetch(`/market/search?q=${encodeURIComponent(query)}`);
}

// ============================================================
// Research / RAG (future endpoints)
// ============================================================

/**
 * Trigger RAG query over ingested documents.
 * POST /research/query
 * Body: { query: string, ticker?: string }
 */
export async function queryResearch(query, ticker = null) {
  return apiFetch('/research/query', {
    method: 'POST',
    body: JSON.stringify({ query, ticker }),
  });
}

/**
 * Upload a PDF for RAG ingestion.
 * POST /research/upload
 */
export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  const url = `${BASE_URL}/research/upload`;
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) throw new Error('Upload failed');
  return response.json();
}

export default {
  healthCheck,
  sendMessage,
  getStockQuote,
  getStockChart,
  searchSecurities,
  queryResearch,
  uploadDocument,
};
