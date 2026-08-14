FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt
COPY src/ src/
COPY artifacts/ artifacts/
# Non-root: the container has no reason to run privileged, and a serving process
# reachable from the network is the last place to leave that lying around.
RUN useradd -m -u 1000 svc && chown -R svc:svc /app
USER svc
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').json()['status']=='ok' else 1)"
CMD ["uvicorn", "src.pipeline.serving:app", "--host", "0.0.0.0", "--port", "8000"]
