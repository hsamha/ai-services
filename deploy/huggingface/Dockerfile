# The chatbot as one Hugging Face Space container.
#
# One process: Streamlit, calling the service's code in process. No FastAPI, no
# Qdrant container -- vectors are in Qdrant Cloud. The upload app is left out,
# so a public visitor can only ask.
#
# Paths are relative to the Space repository's root, where this file is copied
# as `Dockerfile` beside `src/`, `ui/`, `data/` and `requirements.txt`. See
# DEPLOY.md beside this file.
FROM python:3.13-slim

# Spaces run the container as user 1000, so everything is installed and owned
# by that user rather than root.
RUN useradd --create-home --uid 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Dependencies first, so a code change does not reinstall them.
COPY --chown=user requirements.txt ./
COPY --chown=user ui/requirements.txt ./ui/
RUN pip install --no-cache-dir --user -r ui/requirements.txt

COPY --chown=user src ./src
COPY --chown=user ui ./ui
# The constitution's articles and chapters, read by the agent's tools.
COPY --chown=user data ./data

# `streamlit run` puts the script's own folder on the path, not the workdir,
# so neither `ui` nor `src` would import without this.
ENV PYTHONPATH=$HOME/app

# Non-secret configuration for the public demo. The secrets -- QDRANT_CLOUD_URL
# and QDRANT_API_KEY -- are set in the Space's settings, never here.
ENV QDRANT_ENVIRONMENT=cloud \
    OPENAI_MODELS_ONLY=true

# The port a Docker Space serves, matching `app_port` in README.md.
EXPOSE 7860

CMD ["streamlit", "run", "ui/chat_app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
