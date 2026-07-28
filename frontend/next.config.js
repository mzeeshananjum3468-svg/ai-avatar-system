/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Allow access from local network/dev machines
  allowedDevOrigins: [
    'localhost',
    '127.0.0.1',
    '10.28.80.162',
    '10.28.81.139',
  ],

  images: {
    dangerouslyAllowSVG: true,
    unoptimized: process.env.NODE_ENV === 'development',

    // Allow images from any source
    remotePatterns: [
      {
        protocol: 'http',
        hostname: '**',
      },
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
  },

  env: {
    NEXT_PUBLIC_API_URL:
      process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',

    NEXT_PUBLIC_WS_URL:
      process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
  },
};

module.exports = nextConfig;