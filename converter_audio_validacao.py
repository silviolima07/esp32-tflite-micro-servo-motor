import wave
from pathlib import Path
import struct

wav_path = Path("validacao_mfcc/FECHAR_PORTA_1.wav")

cc_path = Path("main/audio/validacao_fechar_01.cc")
h_path = Path("main/audio/validacao_fechar_01.h")

with wave.open(str(wav_path), "rb") as wav:
    canais = wav.getnchannels()
    sample_rate = wav.getframerate()
    sample_width = wav.getsampwidth()
    num_samples = wav.getnframes()
    dados = wav.readframes(num_samples)

print(f"Canais       : {canais}")
print(f"Sample rate  : {sample_rate}")
print(f"Sample width : {sample_width * 8} bits")
print(f"Amostras     : {num_samples}")
print(f"Duracao      : {num_samples / sample_rate:.3f} s")

if canais != 1:
    raise ValueError("O WAV precisa ser mono.")

if sample_rate != 16000:
    raise ValueError("O WAV precisa estar em 16000 Hz.")

if sample_width != 2:
    raise ValueError("O WAV precisa ser PCM16.")

pcm = struct.unpack("<" + "h" * num_samples, dados)

h_path.write_text(
"""#pragma once

#include <cstddef>
#include <cstdint>

extern const int16_t validacao_fechar_01_pcm[];
extern const size_t validacao_fechar_01_num_samples;
""",
encoding="utf-8"
)

with cc_path.open("w", encoding="utf-8") as f:
    f.write('#include "validacao_fechar_01.h"\n\n')

    f.write("const int16_t validacao_fechar_01_pcm[] = {\n")

    for i in range(0, num_samples, 16):
        bloco = pcm[i:i + 16]
        f.write("    " + ", ".join(map(str, bloco)) + ",\n")

    f.write("};\n\n")

    f.write(
        f"const size_t validacao_fechar_01_num_samples = {num_samples};\n"
    )

print()
print("Arquivos gerados:")
print(h_path)
print(cc_path)