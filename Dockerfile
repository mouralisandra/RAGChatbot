FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
