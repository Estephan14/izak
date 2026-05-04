import '../style.css';

export const metadata = {
  title: 'FitCoach AI',
  description: 'AI basketball and weightlifting coach using MediaPipe pose tracking.',
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  userScalable: false,
  viewportFit: 'cover',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
