FROM python:3.13

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

# ✅ Make Streamlit write under /app instead of /.streamlit
ENV HOME=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
RUN mkdir -p /app/.streamlit

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]