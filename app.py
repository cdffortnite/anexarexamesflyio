import os
import base64
import io
import tempfile
import requests
from flask import Flask, request, jsonify, send_file, make_response, send_from_directory
from flask_cors import CORS
from flask_compress import Compress  # Importa a biblioteca de compressão
from werkzeug.utils import secure_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime, timedelta
from collections import defaultdict, deque
from fpdf import FPDF
from uuid import uuid4
from flask import url_for
from pdfminer.high_level import extract_text as pdf_extract_text
from pdf2image import convert_from_bytes
import docx2txt
import pytesseract
from PIL import Image, ImageFilter
import re


def _preprocess_image(image: Image.Image) -> Image.Image:
    """Melhora a qualidade da imagem para o OCR."""
    try:
        osd = pytesseract.image_to_osd(image)
        angle_match = re.search(r"Rotate: (\d+)", osd)
        if angle_match:
            angle = int(angle_match.group(1))
            if angle:
                image = image.rotate(angle, expand=True)
    except Exception:
        pass

    gray = image.convert("L")
    gray = gray.filter(ImageFilter.MedianFilter())
    bw = gray.point(lambda x: 0 if x < 128 else 255, "1")
    return bw
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # Permite conexões de outros domínios (como seu frontend no AwardSpace)
Compress(app)  # Ativa a compressão de respostas

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["50 per minute"]  # Você pode ajustar: "50 per minute", "200 per hour" etc.
)

# Configuração da API DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Pegue a chave da API no Render
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not DEEPSEEK_API_KEY or not OPENAI_API_KEY:
    raise RuntimeError(
        "Defina DEEPSEEK_API_KEY e OPENAI_API_KEY nas variáveis de ambiente"
    )
client = OpenAI()

# Histórico da conversa (poderia ser substituído por um banco de dados se necessário)
user_conversations = {}
# Conversas para interpretacao de exames
user_conversations_exames = {}
# Armazena a análise gerada pela OpenAI para cada usuário
analises_gpt = {}

# Registros para prevenção de spam
user_message_logs = defaultdict(deque)  # armazena mensagens recentes
user_upload_logs = defaultdict(deque)   # armazena hashes dos uploads recentes
blocked_users = {}


def _clean_old(log, window: int = 120):
    """Remove registros antigos de uma deque."""
    now = datetime.utcnow()
    while log and (now - log[0][0]).total_seconds() > window:
        log.popleft()


def _register(logs, user_id: str, content: str) -> int:
    """Registra conteúdo e retorna contagem de repetições no intervalo."""
    queue = logs[user_id]
    now = datetime.utcnow()
    queue.append((now, content))
    _clean_old(queue)
    return sum(1 for _, c in queue if c == content)


def _is_blocked(user_id: str) -> bool:
    expire = blocked_users.get(user_id)
    if expire and expire > datetime.utcnow():
        return True
    if expire:
        del blocked_users[user_id]
    return False


def check_message_spam(user_id: str, content: str, threshold: int = 3) -> bool:
    """Verifica mensagens repetidas e bloqueia temporariamente se necessário."""
    if _is_blocked(user_id):
        return True
    count = _register(user_message_logs, user_id, content)
    if count >= threshold:
        blocked_users[user_id] = datetime.utcnow() + timedelta(minutes=1)
        return True
    return False


def check_upload_spam(user_id: str, upload_hash: str, threshold: int = 3) -> bool:
    """Verifica uploads repetidos."""
    if _is_blocked(user_id):
        return True
    count = _register(user_upload_logs, user_id, upload_hash)
    if count >= threshold:
        blocked_users[user_id] = datetime.utcnow() + timedelta(minutes=1)
        return True
    return False

# Contexto médico especializado para guiar a IA
CONTEXT_MEDICO = (
    "Você é o assistente Sapphir, um chatbot médico projetado para fornecer respostas rápidas e diretas a profissionais de saúde, baseadas em diretrizes clínicas atualizadas e evidências científicas.\n"
    "Priorize velocidade e objetividade. Responda imediatamente sem ajustes de tom ou complexidade.\n"
    "Siga estas diretrizes:\n"
    "1. *Respostas Diretas e Rápidas*: Forneça a resposta de maneira objetiva, sem necessidade de adaptação para diferentes níveis de especialização. Utilize linguagem técnica padrão.\n"
    "2. *Base Científica*: Utilize fontes como PubMed, Cochrane, UpToDate, NICE, WHO e diretrizes médicas reconhecidas. Cite fontes apenas se explicitamente solicitado pelo usuário.\n"
    "3. *Uso de Ferramentas Clínicas*: Quando aplicável, inclua escores e cálculos médicos relevantes (ex.: CHA₂DS₂-VASc, HAS-BLED, SOFA, APACHE II, MELD, Child-Pugh, etc.).\n"
    "4. *Evite Explicações Desnecessárias*: Presuma que o usuário tem conhecimento técnico. Não forneça definições básicas ou contexto introdutório.\n"
    "5. *Tom Profissional e Objetivo*: Sempre responda como um médico experiente, focado na prática clínica.\n"
    "6. *Concisão e Eficiência*: Mantenha as respostas objetivas e, se necessário, utilize até 300 tokens para evitar cortes. Use emojis de forma sutil para tornar a resposta mais fluida, mas sem comprometer a formalidade médica. 😊\n"
)

# Contexto para interpretação de exames
CONTEXT_EXAMES = (
    "Você é um especialista em interpretação de exames laboratoriais e por imagem.\n"
    "Sua função é interpretar resultados clínicos de forma objetiva, rápida e com base científica.\n"
    "Nunca peça o laudo novamente, ele já foi enviado no início da conversa.\n"
    "Use linguagem médica direta. Siga estrutura médica e destaque achados anormais.\n"
)


# Banco de respostas rápidas para perguntas comuns
RESPOSTAS_PADRAO = {
    "quais são os sintomas de dengue?": "Os sintomas da dengue incluem febre alta, dores musculares, dor atrás dos olhos, manchas vermelhas na pele e fadiga intensa. Se houver sinais de gravidade, como sangramento ou tontura intensa, procure atendimento médico imediato.",
    "como tratar uma gripe?": "O tratamento da gripe inclui repouso, hidratação e uso de antitérmicos para febre. Se houver falta de ar ou sintomas persistentes, consulte um médico.",
    "quando tomar antibiótico?": "Antibióticos devem ser usados somente com prescrição médica para infecções bacterianas. O uso inadequado pode causar resistência aos medicamentos."
}


def obter_laudo_openai(conteudo: str) -> str:
    """Gera um laudo técnico e objetivo utilizando o GPT-4o."""
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um médico especialista. Leia o conteúdo abaixo e gere um "
                "laudo clínico técnico e direto, resumindo achados relevantes e "
                "condutas possíveis, com linguagem médica. Ignore ruídos ou "
                "palavras fora de contexto. Caso o texto esteja ilegível ou "
                "incompleto, retorne de forma educada, mas sem suposições "
                "clínicas. Mantenha a resposta concisa, com limite máximo de 300 tokens."
            ),
        },
        {"role": "user", "content": conteudo},
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError("Erro ao consultar a OpenAI.") from exc

def obter_resposta_deepseek(mensagem: str, user_id: str) -> str:
    """Envia uma mensagem para o modelo DeepSeek mantendo o histórico."""
    if user_id not in user_conversations:
        user_conversations[user_id] = [{"role": "system", "content": CONTEXT_MEDICO}]
        analise = analises_gpt.pop(user_id, None)
        if analise:
            user_conversations[user_id].append({"role": "system", "content": analise})

    user_conversations[user_id].append({"role": "user", "content": mensagem})

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "deepseek-chat",
        "temperature": 0.2,
        "max_tokens": 300,
        "messages": user_conversations[user_id],
    }

    response = requests.post(DEEPSEEK_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Erro na API DeepSeek: {response.status_code}"

    deepseek_resp = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    user_conversations[user_id].append({"role": "assistant", "content": deepseek_resp})
    return deepseek_resp


def obter_resposta_deepseek_exames(mensagem: str, user_id: str) -> str:
    """Envio de mensagens para o DeepSeek no contexto de exames."""
    if user_id not in user_conversations_exames:
        user_conversations_exames[user_id] = [{"role": "system", "content": CONTEXT_EXAMES}]

    user_conversations_exames[user_id].append({"role": "user", "content": mensagem})

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "deepseek-chat",
        "temperature": 0.2,
        "max_tokens": 300,
        "messages": user_conversations_exames[user_id],
    }

    response = requests.post(DEEPSEEK_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Erro na API DeepSeek: {response.status_code}"

    content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    user_conversations_exames[user_id].append({"role": "assistant", "content": content})
    return content


def enviar_laudo_para_deepseek(laudo: str, user_id: str) -> str:
    """Encaminha o laudo extraído via OpenAI para o contexto de exames."""
    return obter_resposta_deepseek_exames(laudo, user_id)


def extrair_texto(file_storage) -> str:
    """Extrai texto de diversos formatos de arquivo."""
    filename = secure_filename(file_storage.filename)
    ext = os.path.splitext(filename)[1].lower()
    data = file_storage.read()
    file_storage.seek(0)

    try:
        if ext == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                tmp.flush()
                texto = pdf_extract_text(tmp.name)
                pdf_path = tmp.name
            if not texto.strip() or len(texto.strip()) < 30:
                try:
                    images = convert_from_bytes(data, dpi=300)
                    ocr_result = []
                    for img in images:
                        processed = _preprocess_image(img)
                        ocr_result.append(
                            pytesseract.image_to_string(
                                processed, config="--oem 3 --psm 6 -l por+eng"
                            )
                        )
                    texto = "\n".join(ocr_result)
                except Exception:
                    texto = ""
            os.unlink(pdf_path)
            return texto
        elif ext in [".doc", ".docx"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(data)
                tmp.flush()
                texto = docx2txt.process(tmp.name)
            os.unlink(tmp.name)
            return texto
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]:
            try:
                b64 = base64.b64encode(data).decode("utf-8")
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcreva todo o texto presente na imagem:"},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"}
                    ],
                }]
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=1024,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                try:
                    image = Image.open(io.BytesIO(data))
                    processed = _preprocess_image(image)
                    return pytesseract.image_to_string(processed, config="--oem 3 --psm 6 -l por")
                except Exception:
                    return ""
        else:
            return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

@app.route("/")
def home():
    """Verifica se a API está rodando corretamente"""
    return jsonify({"message": "API do DeepSeek rodando!"})


@app.route("/chatbot")
def chatbot_page():
    """Serve a interface web do chatbot."""
    return send_file(os.path.join(app.root_path, "anexarexames.html"))

@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    """Mantém histórico da conversa e retorna uma resposta da API DeepSeek."""
    # Aceita JSON ou application/x-www-form-urlencoded
    data = request.get_json(silent=True) or request.form or {}
    user_message = data.get("message", "").strip()
    user_id = data.get("user_id") or request.remote_addr

    if not user_message:
        return jsonify({"error": "Nenhuma mensagem recebida."}), 400

    if check_message_spam(user_id, user_message):
        return jsonify({"error": "Mensagens repetidas detectadas."}), 429

    # Se a mensagem já tem uma resposta rápida no banco de dados
    if user_message.lower() in RESPOSTAS_PADRAO:
        return jsonify({"response": RESPOSTAS_PADRAO[user_message.lower()]})

    resposta = obter_resposta_deepseek(user_message, user_id)
    return jsonify({"response": resposta})


@app.route("/chat_exames", methods=["POST"])
@limiter.limit("5 per minute")
def chat_exames():
    """Chat especializado que utiliza o laudo de exames como contexto."""
    data = request.json
    user_message = data.get("message", "").strip()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id obrigatório."}), 400

    if not user_message:
        return jsonify({"error": "Nenhuma mensagem recebida."}), 400

    if check_message_spam(user_id, user_message):
        return jsonify({"error": "Mensagens repetidas detectadas."}), 429

    resposta = obter_resposta_deepseek_exames(user_message, user_id)
    return jsonify({"response": resposta})

@app.route("/upload", methods=["POST"])
@limiter.limit("3 per minute")
def upload_exames():
    if request.content_type and not request.content_type.startswith("multipart/form-data"):
        return jsonify({"error": "Content-Type deve ser multipart/form-data"}), 400
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id obrigatório."}), 400
    arquivos = request.files.getlist("file")
    textos = []

    for arquivo in arquivos:
        filename = secure_filename(arquivo.filename)
        texto = extrair_texto(arquivo)

        if texto.strip():
            textos.append(f"\n📄 {filename}:\n{texto.strip()}\n")
        else:
            textos.append(f"\n📄 {filename}:\n[Não foi possível extrair texto do arquivo]\n")

    conteudo_total = "".join(textos) if textos else "[Nenhum texto extra\u00eddo dos exames]"
    conteudo_hash = str(hash(conteudo_total))

    if check_upload_spam(user_id, conteudo_hash):
        return jsonify({"error": "Envio repetido detectado."}), 429

    if not conteudo_total.strip() or len(conteudo_total.strip()) < 30:
        mensagem = (
            "Texto do exame insuficiente para gerar um laudo. "
            "Verifique a qualidade da imagem enviada."
        )
        analises_gpt[user_id] = mensagem
        return jsonify({"laudo": mensagem})

    prompt = (
        "Analise os seguintes exames médicos e forneça um laudo resumido com condutas objetivas:\n"
        f"{conteudo_total}"
    )

    try:
        laudo = obter_laudo_openai(prompt)
    except Exception:
        return jsonify({"error": "Erro ao consultar a OpenAI."}), 500

    analises_gpt[user_id] = laudo

    # Envia automaticamente o laudo para o DeepSeek
    enviar_laudo_para_deepseek(laudo, user_id)

    # Retorna o laudo gerado para que o frontend possa exibi-lo
    return jsonify({"status": "success", "laudo": laudo})


@app.route("/download-laudo", methods=["POST"])
def download_laudo():
    """Gera um PDF a partir do laudo enviado e retorna a URL do arquivo."""
    data = request.get_json(silent=True) or {}
    laudo = data.get("laudo", "").strip()

    if not laudo:
        return jsonify({"error": "Nenhum laudo recebido."}), 400

    os.makedirs(os.path.join(app.root_path, "static", "pdfs"), exist_ok=True)
    filename = f"laudo_{uuid4().hex}.pdf"
    filepath = os.path.join(app.root_path, "static", "pdfs", filename)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Laudo do Exame", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for line in laudo.splitlines():
        pdf.multi_cell(0, 8, line)
        pdf.ln(1)
    pdf.output(filepath)

    pdf_url = url_for("static", filename=f"pdfs/{filename}", _external=True)
    return jsonify({"pdf_url": pdf_url})


@app.route("/laudo_pdf", methods=["GET"])
def laudo_pdf():
    """Gera um PDF com o último laudo do usuário."""
    user_id = request.args.get("user_id", "default_user")
    laudo = analises_gpt.get(user_id)
    if not laudo:
        return "Nenhum laudo disponível", 404

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Laudo do Exame", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for line in laudo.splitlines():
        pdf.multi_cell(0, 8, line)
        pdf.ln(1)

    pdf_output = pdf.output(dest="S").encode("latin-1")
    response = make_response(pdf_output)
    response.headers.set("Content-Type", "application/pdf")
    response.headers.set("Content-Disposition", "attachment", filename="laudo.pdf")
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Usa a variável de ambiente PORT ou 5000 como valor padrão
    app.run(host="0.0.0.0", port=port) 
