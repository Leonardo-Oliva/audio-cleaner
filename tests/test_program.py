import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Adiciona o diretório principal ao Python Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Program import app  # Importa a aplicação FastAPI do arquivo Program.py

client = TestClient(app)


@pytest.fixture
def mock_firebase(mocker):
    # Simula as credenciais do Firebase
    mocker.patch("Program.credentials.Certificate", return_value=MagicMock())
    mocker.patch("Program.firebase_admin.initialize_app")
    mocker.patch("Program.storage.bucket", return_value=MagicMock())


@pytest.fixture
def mock_audio_processing(mocker):
    # Simula o processamento de áudio
    mocker.patch("Program.AudioFile")
    mocker.patch("Program.nr.reduce_noise", return_value=[[0.1] * 100])  # Áudio simulado
    mocker.patch("Program.Pedalboard", return_value=MagicMock())


@pytest.fixture
def mock_file_system(mocker):
    # Simula o sistema de arquivos
    mocker.patch("os.makedirs")
    mocker.patch("os.remove")
    mocker.patch("os.path.exists", return_value=True)


#def test_process_audio(mock_firebase, mock_audio_processing, mock_file_system):
    # Simula o envio de um arquivo de áudio
    mock_upload = MagicMock()
    mock_upload.read.return_value = b"dummy_audio_data"

    # Envia o arquivo para o endpoint
    response = client.post(
        "/process_audio/",
        files={"file": ("test.wav", b"fake audio content")},
        data={"user_id": "test_user"}
    )

    # Verifica se a resposta está correta
    assert response.status_code == 200
    assert response.json() == {"message": "Áudio processado com sucesso!"}


def test_process_audio_invalid_file(mock_firebase, mock_audio_processing, mock_file_system):
    # Testa o envio de um arquivo inválido
    response = client.post(
        "/process_audio/",
        files={"file": ("test.txt", b"fake audio content")},
        data={"user_id": "test_user"}
    )

    # Verifica se o erro é retornado corretamente
    assert response.status_code == 400
    assert response.json()["detail"] == "Apenas arquivos .wav são permitidos."
