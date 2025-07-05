/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: "http://localhost:8000/:path*",
      },
    ];
  },
  env: {
    NEXT_PUBLIC_API_URL:
      process.env.NODE_ENV === "production"
        ? "https://your-backend-url.com"
        : "http://localhost:8000",
  },
};

module.exports = nextConfig;
