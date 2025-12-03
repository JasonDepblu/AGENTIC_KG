/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // ChatGPT-inspired dark theme
        'bg-primary': '#212121',
        'bg-secondary': '#2f2f2f',
        'bg-sidebar': '#171717',
        'bg-hover': '#424242',
        'bg-input': '#2f2f2f',
        'text-primary': '#ececec',
        'text-secondary': '#8e8e8e',
        'accent': '#10a37f',
        'accent-hover': '#0d8a6a',
        'border': '#424242',
        'border-light': '#5a5a5a',
      },
    },
  },
  plugins: [],
}
