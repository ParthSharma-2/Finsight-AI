/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'Consolas', 'monospace'],
        display: ['"Syne"', 'sans-serif'],
        body: ['"DM Sans"', 'sans-serif'],
      },
      colors: {
        terminal: {
          bg: '#080c10',
          surface: '#0d1117',
          panel: '#111820',
          border: '#1e2d3d',
          accent: '#00d4ff',
          green: '#00ff88',
          amber: '#ffb800',
          red: '#ff4466',
          purple: '#9f7aea',
          muted: '#4a6278',
          text: '#c9d8e8',
          dim: '#6e8899',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'scan': 'scan 4s linear infinite',
        'flicker': 'flicker 8s ease-in-out infinite',
        'ticker': 'ticker 40s linear infinite',
        'blink': 'blink 1.2s step-end infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'slide-up': 'slideUp 0.4s ease-out',
        'fade-in': 'fadeIn 0.6s ease-out',
      },
      keyframes: {
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        flicker: {
          '0%, 100%': { opacity: 1 },
          '92%': { opacity: 1 },
          '93%': { opacity: 0.85 },
          '94%': { opacity: 1 },
          '96%': { opacity: 0.9 },
          '97%': { opacity: 1 },
        },
        ticker: {
          '0%': { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-50%)' },
        },
        blink: {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0 },
        },
        glow: {
          '0%': { textShadow: '0 0 10px #00d4ff, 0 0 20px #00d4ff40' },
          '100%': { textShadow: '0 0 20px #00d4ff, 0 0 40px #00d4ff60, 0 0 60px #00d4ff20' },
        },
        slideUp: {
          '0%': { opacity: 0, transform: 'translateY(12px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%': { opacity: 0 },
          '100%': { opacity: 1 },
        },
      },
      boxShadow: {
        'terminal': '0 0 0 1px #1e2d3d, 0 4px 32px rgba(0,0,0,0.6)',
        'glow-cyan': '0 0 20px rgba(0,212,255,0.3)',
        'glow-green': '0 0 20px rgba(0,255,136,0.3)',
        'inner-terminal': 'inset 0 1px 0 #1e2d3d',
      }
    },
  },
  plugins: [],
}
