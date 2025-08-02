FROM debian:bookworm-slim

# Dependências para compilar Python
RUN apt-get update && apt-get install -y \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Força o mise a compilar o Python
ENV MISE_PYTHON_COMPILE=1

# Instala o mise
RUN curl -fsSL https://mise.jdx.dev/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"

# Compila e define o Python 3.11 como padrão
RUN mise install python@3.11 && mise global python@3.11

# Diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . /app

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Expondo a porta utilizada pelo Fly.io
EXPOSE 8080

# Comando de inicialização
CMD ["bash", "-c", "gunicorn -w 6 --threads 2 -b 0.0.0.0:$PORT app:app"]
