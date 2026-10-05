from pathlib import Path
import wave
import numpy as np


ARQUIVOS = [
    ("Teste_ABRIR_PORTA_01.wav",  "teste_abrir_01"),
    ("Teste_ABRIR_PORTA_02.wav",  "teste_abrir_02"),
    ("Teste_FECHAR_PORTA_01.wav", "teste_fechar_01"),
    ("Teste_FECHAR_PORTA_02.wav", "teste_fechar_02"),
    
      # Novos testes externos
    ("matheus_eleven_fechar_porta.wav", "teste_05"),
    ("igor_audacity_abrir_porta.wav",   "teste_06"),
]

PASTA_WAV = Path("teste_audio")
PASTA_SAIDA = Path("main/audio")

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)


def converter(nome_wav, nome_variavel):

    caminho = PASTA_WAV / nome_wav

    print("=" * 60)
    print(f"Convertendo: {caminho}")

    with wave.open(str(caminho), "rb") as wav:
        canais = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        num_samples = wav.getnframes()

        dados = wav.readframes(num_samples)

    if canais != 1:
        raise ValueError(f"{nome_wav}: esperado áudio mono")

    if sample_rate != 16000:
        raise ValueError(
            f"{nome_wav}: esperado 16000 Hz, encontrado {sample_rate}"
        )

    if sample_width != 2:
        raise ValueError(
            f"{nome_wav}: esperado PCM16"
        )

    pcm = np.frombuffer(dados, dtype="<i2")

    print(f"Sample rate : {sample_rate} Hz")
    print(f"Amostras    : {len(pcm)}")
    print(f"Duração     : {len(pcm) / sample_rate:.3f} s")

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    arquivo_h = PASTA_SAIDA / f"{nome_variavel}.h"

    header = f"""#pragma once

#include <stdint.h>

extern const int16_t {nome_variavel}_pcm[];
extern const int {nome_variavel}_num_samples;
"""

    arquivo_h.write_text(header, encoding="utf-8")

    # ---------------------------------------------------------
    # SOURCE
    # ---------------------------------------------------------

    arquivo_cc = PASTA_SAIDA / f"{nome_variavel}.cc"

    with arquivo_cc.open("w", encoding="utf-8") as f:

        f.write(f'#include "{nome_variavel}.h"\n\n')

        f.write(f"const int16_t {nome_variavel}_pcm[] = {{\n")

        valores_por_linha = 16

        for i in range(0, len(pcm), valores_por_linha):
            bloco = pcm[i:i + valores_por_linha]

            linha = ", ".join(str(int(v)) for v in bloco)

            f.write("    " + linha)

            if i + valores_por_linha < len(pcm):
                f.write(",")

            f.write("\n")

        f.write("};\n\n")

        f.write(
            f"const int {nome_variavel}_num_samples = "
            f"sizeof({nome_variavel}_pcm) / "
            f"sizeof({nome_variavel}_pcm[0]);\n"
        )

    print(f"Gerado: {arquivo_h}")
    print(f"Gerado: {arquivo_cc}")


for nome_wav, nome_variavel in ARQUIVOS:
    converter(nome_wav, nome_variavel)

print()
print("=" * 60)
print("CONVERSAO CONCLUIDA")
print("=" * 60)
print(f"{len(ARQUIVOS)} arquivos WAV convertidos.")