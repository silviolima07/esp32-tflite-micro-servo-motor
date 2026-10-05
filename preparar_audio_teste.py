from pathlib import Path
import librosa
import numpy as np

ARQUIVO = Path("teste_audio/Teste_ABRIR_PORTA_01.wav")

SR = 16000
DURACAO = 2.5
MARGEM = 0.2
TOP_DB = 30

TAMANHO_FINAL = int(SR * DURACAO)   # 40000 amostras
MARGEM_SAMPLES = int(SR * MARGEM)  # 3200 amostras

# --------------------------------------------------
# Carrega o WAV
# --------------------------------------------------

audio, sr = librosa.load(
    ARQUIVO,
    sr=SR,
    mono=True
)

print("==============================================")
print("AUDIO ORIGINAL")
print("==============================================")
print(f"Arquivo          : {ARQUIVO.name}")
print(f"Sample rate      : {sr} Hz")
print(f"Amostras         : {len(audio)}")
print(f"Duracao          : {len(audio) / sr:.3f} s")

# --------------------------------------------------
# Detecta a região que contém áudio/fala
# --------------------------------------------------

# 1. Detecta a fala
_, indice = librosa.effects.trim(
    audio,
    top_db=TOP_DB
)

inicio_fala = indice[0]
fim_fala = indice[1]

# 2. Acrescenta margem de 200 ms
inicio = max(0, inicio_fala - MARGEM_SAMPLES)
fim = min(len(audio), fim_fala + MARGEM_SAMPLES)

# 3. Agora existe o audio_recortado
audio_recortado = audio[inicio:fim]

# 4. Inicializa os valores de padding
antes = 0
depois = 0

# 5. Ajusta para 40.000 amostras
if len(audio_recortado) < TAMANHO_FINAL:

    faltam = TAMANHO_FINAL - len(audio_recortado)

    antes = faltam // 2
    depois = faltam - antes

    audio_final = np.pad(
        audio_recortado,
        (antes, depois),
        mode="constant"
    )

elif len(audio_recortado) > TAMANHO_FINAL:

    excesso = len(audio_recortado) - TAMANHO_FINAL
    inicio_corte = excesso // 2

    audio_final = audio_recortado[
        inicio_corte:inicio_corte + TAMANHO_FINAL
    ]

else:
    audio_final = audio_recortado
    
    
# 6. SOMENTE AGORA mostramos todos os valores
print("\nREFERENCIA PARA IMPLEMENTACAO ESP32")
print("----------------------------------------------")
print(f"Inicio detectado  : {inicio_fala}")
print(f"Fim detectado     : {fim_fala}")
print(f"Margem samples    : {MARGEM_SAMPLES}")
print(f"Inicio com margem : {inicio}")
print(f"Fim com margem    : {fim}")
print(f"Amostras recorte  : {len(audio_recortado)}")
print(f"Padding antes     : {antes}")
print(f"Padding depois    : {depois}")
print(f"Amostras finais   : {len(audio_final)}")

print()
print("REGIAO DETECTADA")
print("----------------------------------------------")
print(f"Inicio           : {inicio_fala} amostras")
print(f"Fim              : {fim_fala} amostras")
print(f"Inicio           : {inicio_fala / sr:.3f} s")
print(f"Fim              : {fim_fala / sr:.3f} s")

# --------------------------------------------------
# Acrescenta 200 ms antes e depois
# --------------------------------------------------

inicio = max(0, inicio_fala - MARGEM_SAMPLES)
fim = min(len(audio), fim_fala + MARGEM_SAMPLES)

audio_recortado = audio[inicio:fim]

print()
print("COM MARGEM DE 200 ms")
print("----------------------------------------------")
print(f"Inicio           : {inicio} amostras")
print(f"Fim              : {fim} amostras")
print(f"Amostras         : {len(audio_recortado)}")
print(f"Duracao          : {len(audio_recortado) / sr:.3f} s")

# --------------------------------------------------
# Ajusta para exatamente 2,5 segundos
# --------------------------------------------------

if len(audio_recortado) < TAMANHO_FINAL:

    faltam = TAMANHO_FINAL - len(audio_recortado)

    antes = faltam // 2
    depois = faltam - antes

    audio_final = np.pad(
        audio_recortado,
        (antes, depois),
        mode="constant"
    )

elif len(audio_recortado) > TAMANHO_FINAL:

    excesso = len(audio_recortado) - TAMANHO_FINAL

    inicio_corte = excesso // 2

    audio_final = audio_recortado[
        inicio_corte:inicio_corte + TAMANHO_FINAL
    ]

else:
    audio_final = audio_recortado

print()
print("AUDIO FINAL")
print("----------------------------------------------")
print(f"Amostras         : {len(audio_final)}")
print(f"Duracao          : {len(audio_final) / sr:.3f} s")
print(f"Esperado         : {TAMANHO_FINAL} amostras")

if len(audio_final) == TAMANHO_FINAL:
    print("Resultado        : OK")
else:
    print("Resultado        : ERRO")