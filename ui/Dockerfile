# The apps, with the service's code beside them: they call it in process.
# Built from the repository root -- see ui/docker-compose.yml.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt ./
COPY ui/requirements.txt ./ui/
RUN pip install --no-cache-dir -r ui/requirements.txt

COPY src ./src
COPY ui ./ui

# `streamlit run` puts the script's own folder on the path, not the workdir,
# so neither `ui` nor `src` would import without this.
ENV PYTHONPATH=/app

EXPOSE 8501 8502
CMD ["streamlit", "run", "ui/upload_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
