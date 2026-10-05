import numpy as np
import librosa
from pathlib import Path

ARQUIVO = Path("validacao_mfcc/FECHAR_PORTA_1.wav")

SR = 16000
DURACAO = 2.5
TARGET_SAMPLES = 40000
MARGEM = 3200

N_MFCC = 13
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 40


# ============================================================
# CARREGAMENTO
# ============================================================

audio, sr = librosa.load(
    ARQUIVO,
    sr=SR,
    mono=True
)

print("Arquivo             :", ARQUIVO.name)
print("Sample rate         :", sr)
print("Amostras originais  :", len(audio))


# ============================================================
# MESMO TRIM USADO NO TREINAMENTO
# ============================================================

_, idx = librosa.effects.trim(
    audio,
    top_db=30
)

speech_start = int(idx[0])
speech_end = int(idx[1])

margin_start = max(0, speech_start - MARGEM)
margin_end = min(len(audio), speech_end + MARGEM)

recorte = audio[margin_start:margin_end]

print()
print("PRE-PROCESSAMENTO PYTHON")
print("Fala inicio         :", speech_start)
print("Fala fim            :", speech_end)
print("Margem inicio       :", margin_start)
print("Margem fim          :", margin_end)
print("Recorte             :", len(recorte))


# ============================================================
# CENTRALIZA EM 40000 AMOSTRAS
# ============================================================

if len(recorte) < TARGET_SAMPLES:

    total_padding = TARGET_SAMPLES - len(recorte)

    padding_before = total_padding // 2
    padding_after = total_padding - padding_before

    audio_final = np.pad(
        recorte,
        (padding_before, padding_after),
        mode="constant"
    )

else:

    inicio = (len(recorte) - TARGET_SAMPLES) // 2

    audio_final = recorte[
        inicio:inicio + TARGET_SAMPLES
    ]

    padding_before = 0
    padding_after = 0


print("Padding antes       :", padding_before)
print("Padding depois      :", padding_after)
print("Amostras finais     :", len(audio_final))

import numpy as np

# ==========================================================
# DIAGNOSTICO FFT - FRAME 75
# ==========================================================

FRAME = 75
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160

start = FRAME * HOP_LENGTH

# O librosa usa uma janela Hann de 400 amostras,
# centralizada dentro da janela FFT de 512.
window = np.hanning(WIN_LENGTH + 1)[:-1]

window_padded = np.zeros(N_FFT, dtype=np.float32)

offset = (N_FFT - WIN_LENGTH) // 2

window_padded[offset:offset + WIN_LENGTH] = window

# Mesmo frame de 512 amostras usado pelo librosa
frame_audio = audio_final[start:start + N_FFT]

windowed = frame_audio * window_padded
# ============================================================
# Diagnostico: entrada da FFT
# ============================================================

print("\n" + "=" * 60)
print("ENTRADA FFT PYTHON - FRAME 75")
print("=" * 60)

indices = [56, 57, 58, 100, 200, 300, 400, 454, 455]

for k in indices:
    print(
        f"fft[{k:03d}] real={windowed[k]: .9f} imag={0.0: .9f}"
    )

# FFT para sinal real
spectrum = np.fft.rfft(windowed, n=N_FFT)

power = np.abs(spectrum) ** 2

print("\n" + "=" * 60)
print("FFT PYTHON - FRAME 75")
print("=" * 60)

for k in range(10):
    print(
        f"bin {k:02d} : "
        f"real={spectrum[k].real: .6f} "
        f"imag={spectrum[k].imag: .6f} "
        f"power={power[k]: .6f}"
    )

# ============================================================
# MFCC - MESMOS PARAMETROS DO TREINAMENTO
# ============================================================

mfcc = librosa.feature.mfcc(
    y=audio_final,
    sr=SR,
    n_mfcc=N_MFCC,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    center=False
)

print()
print("Dimensao MFCC       :", mfcc.shape)


# ============================================================
# MESMOS PONTOS IMPRESSOS PELO ESP32
# ============================================================

frames = [
    0, 25, 50, 75, 100, 125,
    150, 175, 200, 225, 246
]

print()
print("VALORES MFCC - PYTHON/LIBROSA")
print("--------------------------------------------------")

for c in range(N_MFCC):

    print(f"\nMFCC[{c:02d}]")

    for frame in frames:

        print(
            f"  frame {frame:03d} = "
            f"{mfcc[c, frame]:.6f}"
        )