# TEMP
# Base image with PyTorch and required dependencies
FROM nvcr.io/nvidia/pytorch:24.11-py3

# Install additional dependencies
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir \
        transformers==4.38.2 \
        datasets==2.21.0 \
        SentencePiece==0.2.0

WORKDIR /workspace
COPY t5-base.py /workspace/t5-base.py
