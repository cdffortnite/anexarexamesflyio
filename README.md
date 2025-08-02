# Anexar Exames

Backend em Flask para extração e análise de exames médicos com geração de laudos e integração ao chatbot Sapphir.

## Executando localmente

```bash
pip install -r requirements.txt
python -m py_compile app.py
flask run
```

## Deploy no Render
1. Crie um novo **Web Service** e aponte para este repositório.
2. Defina as variáveis de ambiente `DEEPSEEK_API_KEY` e `OPENAI_API_KEY`.
3. O serviço detectará `requirements.txt` e `Procfile` automaticamente.
   ```bash
   gunicorn -w 6 --threads 2 -b 0.0.0.0:$PORT app:app
   ```
4. Opcionalmente utilize o arquivo `render.yaml` para configurar o serviço.

## Endpoints
- `/upload` – envio de exames
- `/chat` – chatbot de anamnese
- `/chat_exames` – discussão do laudo dos exames
- `/download-laudo` – geração de PDF do laudo
- `/chatbot` – página HTML do chatbot

O arquivo `anexarexames.html` pode ser hospedado em qualquer serviço estático (como Awardspace). Ajuste as constantes no final do HTML caso a URL do backend mude.
