FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 HOME=/tmp
RUN pip install --no-cache-dir numpy pandas scikit-learn pillow scipy huggingface-hub nbformat nbclient ipykernel
LABEL helix.runtime="2"
WORKDIR /work
USER 65534:65534
