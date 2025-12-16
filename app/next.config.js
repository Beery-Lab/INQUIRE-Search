/** @type {import('next').NextConfig} */
const nextConfig = {
    rewrites: async () => [
        {
            source: "/api/:path*",
            destination: "http://127.0.0.1:5002/api/:path*", // Proxy to Backend
        },
        {
            source: "/process_query",
            destination: "http://127.0.0.1:5002/process_query", // Proxy to Backend
        },
        {
            source: "/submit_data",
            destination: "http://127.0.0.1:5002/submit_data", // Proxy to Backend
        },
        {
            source: "/dashboard",
            destination: "http://127.0.0.1:5002/dashboard", // Proxy to Backend
        },
    ],
};

module.exports = nextConfig;
