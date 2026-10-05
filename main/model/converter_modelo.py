from pathlib import Path

arquivo_tflite = Path("audacity_modelo_comandos_int8.tflite")
arquivo_h = Path("model.h")
arquivo_cc = Path("model.cc")

dados = arquivo_tflite.read_bytes()

# Gera model.h
arquivo_h.write_text(
    """#pragma once

#include <cstddef>

extern const unsigned char g_model[];
extern const unsigned int g_model_len;
""",
    encoding="utf-8",
)

# Converte bytes para representação hexadecimal
linhas = []

for i in range(0, len(dados), 12):
    bloco = dados[i:i + 12]
    linha = ", ".join(f"0x{byte:02x}" for byte in bloco)
    linhas.append("    " + linha)

conteudo_array = ",\n".join(linhas)

# Gera model.cc
arquivo_cc.write_text(
    f"""#include "model.h"

alignas(16) const unsigned char g_model[] = {{
{conteudo_array}
}};

const unsigned int g_model_len = {len(dados)};
""",
    encoding="utf-8",
)

print("Conversão concluída.")
print(f"Tamanho do modelo: {len(dados):,} bytes")
print(f"Gerado: {arquivo_h}")
print(f"Gerado: {arquivo_cc}")