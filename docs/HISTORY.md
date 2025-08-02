# Histórico de Evolução

Este documento resume as principais mudanças implementadas no sistema de upload e análise de exames.

- **Validação de OCR**: somente arquivos com extração de texto superior a 30 caracteres são processados pelo GPT‑4o.
- **Prompt clínico direto** para interpretação de exames.
- **Chat de exames** (`/chat_exames`) com histórico isolado em `user_conversations_exames`.
- **Rota `/download-laudo`** para gerar PDF do laudo.
- **Frontend atualizado**: botão de upload responsivo, toggle de exibição do laudo e chat flutuante.
- **Rotas limitadas** com `Flask‑Limiter` e exigência de `user_id`.
- **Rota `/chatbot`** para servir a interface HTML.
- **Função `enviar_laudo_para_deepseek`** integra OpenAI e DeepSeek.
- **Layout do chat** do `anexarexames.html` modernizado.
- **Rota `/chat`** aceita usuários sem `user_id`, usando o IP como identificador.

Para executar localmente:

```bash
pip install -r requirements.txt
python -m py_compile app.py
flask run
```
