FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch FIRST, explicitly, before the rest of requirements.txt.
# Without this, pip installs the default CUDA-enabled torch build even on a
# CPU-only host, pulling in nvidia-cublas/cudnn/etc that are pure dead weight
# here -- wastes disk, build time, AND memory headroom on a 512MB instance.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]