FROM node:24-bookworm-slim AS build
WORKDIR /app
COPY package.json package-lock.json tsconfig.json vite.config.mjs index.html ./
RUN npm ci
COPY src ./src
COPY public ./public
RUN npm run build

FROM node:24-bookworm-slim
ENV NODE_ENV=production PORT=3000
WORKDIR /app
COPY hosted/package.json hosted/package-lock.json ./hosted/
RUN cd hosted && npm ci --omit=dev
COPY hosted/*.mjs ./hosted/
COPY local/*.mjs local/*.py ./local/
COPY --from=build /app/dist ./dist
EXPOSE 3000
CMD ["node", "hosted/server.mjs"]
