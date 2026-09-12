import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone puts the server and only the dependencies it actually imports
  // into .next/standalone, so the runtime image carries neither node_modules
  // nor the build toolchain.
  output: "standalone",
};

export default nextConfig;
