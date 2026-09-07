# The apps only. They reach the service over HTTP, so nothing from `src` is
# needed here and this image knows nothing about it.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./ui

# `streamlit run` puts the script's own folder on the path, not the workdir,
# so `ui` would not import without this.
ENV PYTHONPATH=/app

EXPOSE 8501 8502
CMD ["streamlit", "run", "ui/upload_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
