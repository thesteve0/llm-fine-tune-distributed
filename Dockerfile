# Use RHT official PyTorch image as a base
FROM quay.io/modh/training:py311-cuda124-torch251

RUN echo "----> This is my custom log message - definitely new"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && pip install git+https://github.com/huggingface/transformers.git && pip install aim

COPY data/qa_dataset.parquet data/

COPY training.py .

CMD ["python", "training.py"]