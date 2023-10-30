# app/Dockerfile

FROM python:3.10.6-slim

WORKDIR /mod2_simula_incidentes

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ecestebanjek/suanet_modulo_simula_incidentes.git .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app_sim_inc.py", "--server.port=8501", "--server.address=0.0.0.0"]