# TNGGalaxyLab Docker image
#
# Build:
#     docker build -t tnggalaxylab:latest .
#
# Run interactively:
#     docker run -it --rm -v $(pwd):/workspace tnggalaxylab:latest bash
#
# Run validation:
#     docker run --rm tnggalaxylab:latest python validate_synthetic.py \
#         --output /tmp/validation --n-values 1000 10000 100000 --replicas 3
#
# This image provides a fully reproducible environment for running the
# TNGGalaxyLab pipeline on any system supporting Docker.

FROM python:3.12-slim

LABEL maintainer="Abbos Omonov <abbos.omonov@nuu.uz>"
LABEL description="TNGGalaxyLab: reproducible Fourier pipeline for simulated disc galaxies"
LABEL version="1.0"

# System dependencies: HDF5 needs libhdf5-dev; matplotlib needs some font packs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    libgomp1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (pin to tested versions)
WORKDIR /opt/tnggalaxylab
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY tnggalaxylab/ ./tnggalaxylab/
COPY analyze_tng50_subhalo.py .
COPY validate_synthetic.py .
COPY batch_tng50/ ./batch_tng50/

# Non-root user (safer for shared systems)
RUN useradd -m -u 1000 tng && chown -R tng:tng /opt/tnggalaxylab
USER tng

# Environment: make tnggalaxylab importable
ENV PYTHONPATH=/opt/tnggalaxylab
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /workspace

# Default command: show help
CMD ["python", "-c", "import tnggalaxylab; print('TNGGalaxyLab ready. See docs at https://github.com/omonov-abbos/TNGGalaxyLab')"]
