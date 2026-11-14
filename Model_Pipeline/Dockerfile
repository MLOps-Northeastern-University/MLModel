# The model pipeline's image: Python and the pinned libraries, nothing else.
# docker-compose mounts the repo over /app/Model_Pipeline, so code changes need
# no rebuild; the copied code is for running the image on its own (CI, Cloud).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/Model_Pipeline
COPY requirements.txt .
# PyTorch's CPU build first: the default Linux build pulls in GBs of CUDA it can't use here.
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu  && pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "-m", "scripts.run_model_pipeline"]
