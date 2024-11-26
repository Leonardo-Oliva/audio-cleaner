import os
import numpy as np
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pedalboard.io import AudioFile
from pedalboard import NoiseGate, Compressor, LowShelfFilter, Gain, Pedalboard
import noisereduce as nr
import firebase_admin
from firebase_admin import credentials, storage
import uuid

# Configuração do Firebase
firebase_credentials = json.loads(os.environ["FIREBASE_CREDENTIALS"])
cred = credentials.Certificate(firebase_credentials)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'audiocleaner-5dcff.appspot.com'
})

# Definir a taxa de amostragem (sample rate)
sr = 44100

app = FastAPI()

# Adicione as origens permitidas aqui
origins = [
    "http://localhost:3000",  # Frontend em execução, ajuste conforme necessário
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_audio/")
async def process_audio(user_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Apenas arquivos .wav são permitidos.")

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

    # Configurar a pedalboard com vários efeitos
    board = Pedalboard([
        NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
        Compressor(threshold_db=-16, ratio=2.5),
        LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
        Gain(gain_db=10)
    ])

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
    blob = bucket.blob(f'{user_id}/{os.path.basename(output_file_path)}')

    try:
        blob.upload_from_filename(output_file_path)
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
