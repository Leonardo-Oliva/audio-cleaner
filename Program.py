import os
import numpy as np
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pedalboard.io import AudioFile
from pedalboard import NoiseGate, Compressor, LowShelfFilter, Gain, Pedalboard, Reverb, Chorus
import noisereduce as nr
import firebase_admin
from firebase_admin import credentials, storage
import uuid
from pydantic import BaseModel

# Configuração do Firebase
firebase_credentials = json.loads(os.environ["FIREBASE_CREDENTIALS"])
cred = credentials.Certificate(firebase_credentials)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'audiocleaner-5dcff.appspot.com'
})

# Definir a taxa de amostragem (sample rate)
sr = 44100

class AudioProcessParams(BaseModel):
    user_id: str

app = FastAPI()

# Adicione as origens permitidas aqui
origins = [
    "http://localhost:3000",  # Frontend em execução, ajuste conforme necessário
    "http://localhost:3000/upload",
    "https://site-audio-cleaner.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_audio/")
async def process_audio(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    apply_noise_gate: bool = Form(default=True),
    apply_compressor: bool = Form(default=True),
    apply_low_shelf_filter: bool = Form(default=True),
    apply_gain: bool = Form(default=True),
    apply_reverb: bool = Form(default=True),
    apply_chorus: bool = Form(default=True)
):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Apenas arquivos .wav são permitidos.")

    #Cria variavel do nome da pasta
    pasta_efeitos = []

    # Salvar o arquivo de áudio enviado
    os.makedirs('audios', exist_ok=True)

    # Extrair o nome base do arquivo para evitar caminhos maliciosos
    original_filename = os.path.basename(file.filename)
    safe_filename = f"{uuid.uuid4()}_{original_filename}"  # Prefixar com UUID para evitar conflitos
    input_file_path = os.path.join('audios', safe_filename)
    output_file_path = os.path.join('audios', f"{os.path.splitext(original_filename)[0]}_enhanced.wav")

    with open(input_file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Abrir o arquivo de áudio e resamplear para a taxa de amostragem desejada
    try:
        with AudioFile(input_file_path).resampled_to(sr) as f:
            audio = f.read(f.frames)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro no processamento do áudio: {str(e)}")

    # Verificar o número de canais do áudio original
    if audio.ndim == 1:  # Se o áudio for mono (1 canal)
        audio = np.expand_dims(audio, axis=0)  # Adicionar um eixo para torná-lo 2D (1, N)
    elif audio.ndim == 2 and audio.shape[0] > 2:  # Caso o áudio tenha mais de 2 canais
        raise ValueError("O áudio tem mais de 2 canais, o que não é suportado.")

    # Reduzir o ruído usando a biblioteca noisereduce
    reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)

    # Configurar a pedalboard com efeitos baseados nas opções fornecidas
    effects = []
    if apply_noise_gate:
        effects.append(NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250))
        pasta_efeitos.append("noise_gate")
    if apply_compressor:
        effects.append(Compressor(threshold_db=-16, ratio=2.5))
        pasta_efeitos.append("compressor")
    if apply_low_shelf_filter:
        effects.append(LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1))
        pasta_efeitos.append("low_shelf_filter")
    if apply_gain:
        effects.append(Gain(gain_db=10))
        pasta_efeitos.append("gain")
    if apply_reverb:
        effects.append(Reverb(room_size=0.5, damping=0.5, wet_level=0.3))
        pasta_efeitos.append("reverb")
    if apply_chorus:
        effects.append(Chorus(rate_hz=1.5, depth=0.5, mix=0.3))
        pasta_efeitos.append("chorus")

    board = Pedalboard(effects)

    # Aplicar os efeitos ao áudio com ruído reduzido
    effected = board(reduced_noise, sr)

    # Verificar o número de canais do áudio processado
    if effected.ndim == 1:  # Se o áudio for mono (1 canal)
        effected = np.expand_dims(effected, axis=0)  # Torná-lo estéreo (2D)

    # Salvar o áudio processado em um novo arquivo
    with AudioFile(output_file_path, 'w', sr, effected.shape[0]) as f:
        f.write(effected)

    # Enviar o arquivo processado para o Firebase Storage
    bucket = storage.bucket()
    #blob = bucket.blob(f'{user_id}/{os.path.basename(output_file_path)}')
    blob = bucket.blob(f'{user_id}/{"-".join(pasta_efeitos)}+{os.path.basename(output_file_path.replace("_enhanced", ""))}/{os.path.basename(output_file_path)}')
    blob_input_file = bucket.blob(f'{user_id}/{"-".join(pasta_efeitos)}+{os.path.basename(output_file_path.replace("_enhanced", ""))}/{os.path.basename(output_file_path.replace("_enhanced", ""))}')

    try:
        blob.upload_from_filename(output_file_path)
        blob_input_file.upload_from_filename(input_file_path)
        return JSONResponse(content={"message": "Áudio processado com sucesso!"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao enviar para o Firebase: {str(e)}")
    finally:
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
        if os.path.exists(output_file_path):
            os.remove(output_file_path)



@app.post("/process_audio/")
def index():
    return {"name": "AudioCleaner"}
