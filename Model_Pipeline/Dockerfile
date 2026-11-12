# The model pipeline's image: Python and the pinned libraries, nothing else.
# docker-compose mounts the repo over /app/Model_Pipeline, so code changes need
# no rebuild; the copied code is for running the image on its own (CI, Cloud).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/Model_Pipeline
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "-m", "scripts.run_model_pipeline"]
