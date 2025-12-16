/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
    fontFamily: {
      sans: ["Inter var, sans-serif"]
    }
  },
  plugins: [],
  safelist: [
    'bg-slate-800',
    'bg-green-800',
    'bg-red-900'
  ]
}
