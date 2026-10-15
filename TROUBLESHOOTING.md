# Current setup and troubleshooting

See [README](README.md) for verified setup, training, APIs and frontend commands.
GET /health returns 200 only when a compatible model is loaded. If unavailable,
train in the pinned environment and check MODEL_PATH. Build with npm run build.
Existing reports are immutable; use a new run/report directory for reproduction.
A fixed accuracy value does not prove server health.
