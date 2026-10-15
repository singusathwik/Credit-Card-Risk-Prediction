FROM node:22-bookworm-slim AS frontend
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY index.html vite.config.js postcss.config.js tailwind.config.js ./
COPY src ./src
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY credit_risk ./credit_risk
COPY server.py ./
COPY reports/run-001 ./reports/run-001
COPY --from=frontend /app/dist ./dist
RUN useradd --create-home appuser
USER appuser
ENV HOST=0.0.0.0 PORT=5000 MODEL_PATH=/app/artifacts/run-001/model.joblib
EXPOSE 5000
CMD ["python", "server.py"]
