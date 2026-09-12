# The Next.js chat interface.
#
# Three stages: install, build, run. The runtime image gets only the standalone
# server output, so it carries neither node_modules nor the build toolchain.

FROM node:22-alpine AS deps

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci


FROM node:22-alpine AS builder

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Baked in at build time, not read at runtime: NEXT_PUBLIC_* values are
# substituted into the client bundle, which runs in the reader's browser and
# therefore needs an address reachable from there - not from inside the network
# the containers share.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build


FROM node:22-alpine AS runtime

ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

WORKDIR /app

# The standalone bundle carries its own server.js and the dependencies it
# actually imports; the static assets are copied beside it because the server
# reads them from disk rather than bundling them. There is no public/ directory
# in this project - the favicon lives in app/ and is emitted by the build.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

USER node

EXPOSE 3000

CMD ["node", "server.js"]
