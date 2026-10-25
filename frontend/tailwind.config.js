/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // V8 Liquid Glass Semantic Palette
        canvas:   '#07090E',
        surface:  'rgba(18,24,38,0.55)',
        // Electric Phosphor Cyan
        cyan: {
          DEFAULT: '#00F0FF',
          glow:    'rgba(0,240,255,0.20)',
          faint:   'rgba(0,240,255,0.08)',
        },
        // Vivid Mint
        mint: {
          DEFAULT: '#00F5A0',
          glow:    'rgba(0,245,160,0.20)',
          faint:   'rgba(0,245,160,0.08)',
        },
        // Fluid Amber (replaces hard red for debarred/warning)
        amber: {
          DEFAULT: '#FFB020',
          glow:    'rgba(255,176,32,0.20)',
          faint:   'rgba(255,176,32,0.08)',
        },
        // Muted slate
        muted: '#4A5568',
        ghost: 'rgba(255,255,255,0.06)',
      },
      fontFamily: {
        sans:  ['Inter', 'ui-sans-serif', 'system-ui'],
        mono:  ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      backdropBlur: {
        glass: '40px',
      },
      boxShadow: {
        glass:  'inset 0 1px 1px 0 rgba(255,255,255,0.18), 0 20px 40px -15px rgba(0,0,0,0.65)',
        glow:   '0 0 20px rgba(0,240,255,0.25)',
        'glow-mint': '0 0 20px rgba(0,245,160,0.25)',
      },
      borderColor: {
        glass: 'rgba(255,255,255,0.08)',
        'glass-hi': 'rgba(255,255,255,0.14)',
      },
      animation: {
        'ping-slow': 'ping 2.4s cubic-bezier(0,0,0.2,1) infinite',
        'fade-in':   'fadeIn 0.3s ease both',
        'slide-up':  'slideUp 0.4s cubic-bezier(0.16,1,0.3,1) both',
      },
      keyframes: {
        fadeIn:  { from: { opacity: 0 }, to: { opacity: 1 } },
        slideUp: { from: { opacity: 0, transform: 'translateY(30px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
      },
      gridTemplateColumns: {
        layout: 'minmax(0,4fr) minmax(0,5fr) minmax(0,3fr)',
      },
    },
  },
  plugins: [],
}
