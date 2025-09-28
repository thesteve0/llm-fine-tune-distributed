# Use RHT official PyTorch image as a base
FROM quay.io/modh/training:py311-cuda124-torch251

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && pip install git+https://github.com/huggingface/transformers.git

COPY data/qa_dataset.parquet data/

COPY finetune_lyrics.py .

CMD ["python", "finetune_lyrics.py"]