import os
import warnings

# Reduz mensagens informativas do TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Ignora warnings do Python
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np

import librosa
import tensorflow as tf


# ============================================================
# CONFIGURACAO
# ============================================================

AUDIO_DIR = Path("teste_audio")
MODEL_PATH = Path("main/model/audacity_modelo_comandos_int8.tflite")
NORMALIZACAO_PATH = Path(
    "mfcc_npz/audacity_normalizacao_mfcc.npz"
)

SR = 16000
DURACAO = 2.5
MARGEM = 0.2
TOP_DB = 30

N_MFCC = 13
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 40

TAMANHO_FINAL = int(SR * DURACAO)       # 40000
MARGEM_SAMPLES = int(SR * MARGEM)       # 3200


# ============================================================
# CARREGA NORMALIZACAO
# ============================================================

norm = np.load(NORMALIZACAO_PATH)

media = float(norm["media"])
desvio = float(norm["desvio"])

print("=" * 78)
print("VALIDACAO - AUDIOS INEDITOS / MODELO INT8")
print("=" * 78)

print("\nNORMALIZACAO DO TREINAMENTO")
print("-" * 78)
print(f"Media             : {media:.6f}")
print(f"Desvio            : {desvio:.6f}")


# ============================================================
# CARREGA MODELO INT8
# ============================================================

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero = input_details["quantization"]
output_scale, output_zero = output_details["quantization"]

print("\nQUANTIZACAO DO MODELO")
print("-" * 78)
print(f"Input scale       : {input_scale}")
print(f"Input zero point  : {input_zero}")
print(f"Output scale      : {output_scale}")
print(f"Output zero point : {output_zero}")


# ============================================================
# FUNCAO DE PRE-PROCESSAMENTO
# ============================================================

def preparar_audio(caminho):

    audio, sr = librosa.load(
        caminho,
        sr=SR,
        mono=True
    )

    # --------------------------------------------------------
    # Detecta regiao de fala
    # --------------------------------------------------------

    _, indice = librosa.effects.trim(
        audio,
        top_db=TOP_DB
    )

    inicio_fala = indice[0]
    fim_fala = indice[1]

    # --------------------------------------------------------
    # Adiciona margem de 200 ms
    # --------------------------------------------------------

    inicio = max(
        0,
        inicio_fala - MARGEM_SAMPLES
    )

    fim = min(
        len(audio),
        fim_fala + MARGEM_SAMPLES
    )

    audio_recortado = audio[inicio:fim]

    # --------------------------------------------------------
    # Ajusta para exatamente 2.5 segundos
    # --------------------------------------------------------

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
            inicio_corte:
            inicio_corte + TAMANHO_FINAL
        ]

    else:

        audio_final = audio_recortado

    return (
        audio_final,
        len(audio),
        inicio_fala,
        fim_fala
    )


# ============================================================
# FUNCAO DE INFERENCIA
# ============================================================

def inferir(caminho):

    (
        audio_final,
        tamanho_original,
        inicio_fala,
        fim_fala
    ) = preparar_audio(caminho)

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

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
    
    print("\nREFERENCIA MFCC - PYTHON")
    print("----------------------------------------------")
    print("Arquivo:", caminho)
    print("Shape:", mfcc.shape)

    print("\nPrimeiro frame - 13 coeficientes:")
    for i in range(13):
        print(f"MFCC[{i}, 0] = {mfcc[i, 0]:.6f}")

    print("\nAlguns pontos de controle:")
    print(f"MFCC[0, 100]  = {mfcc[0, 100]:.6f}")
    print(f"MFCC[5, 100]  = {mfcc[5, 100]:.6f}")
    print(f"MFCC[12, 246] = {mfcc[12, 246]:.6f}")

    if mfcc.shape != (13, 247):

        raise ValueError(
            f"MFCC inesperado: {mfcc.shape}"
        )

    # --------------------------------------------------------
    # Normalizacao
    # --------------------------------------------------------

    mfcc_normalizado = (
        mfcc - media
    ) / desvio

    # --------------------------------------------------------
    # Tensor (1, 13, 247, 1)
    # --------------------------------------------------------

    entrada_float = mfcc_normalizado[
        np.newaxis,
        ...,
        np.newaxis
    ].astype(np.float32)

    # --------------------------------------------------------
    # Quantizacao INT8
    # --------------------------------------------------------

    entrada_int8 = np.round(
        entrada_float / input_scale
        + input_zero
    )

    entrada_int8 = np.clip(
        entrada_int8,
        -128,
        127
    ).astype(np.int8)

    # --------------------------------------------------------
    # Inferencia
    # --------------------------------------------------------

    interpreter.set_tensor(
        input_details["index"],
        entrada_int8
    )

    interpreter.invoke()

    saida_int8 = interpreter.get_tensor(
        output_details["index"]
    )[0][0]

    # --------------------------------------------------------
    # Desquantizacao
    # --------------------------------------------------------

    prob_fechar = (
        int(saida_int8) - output_zero
    ) * output_scale

    prob_fechar = float(prob_fechar)

    # Protecao numerica
    prob_fechar = max(
        0.0,
        min(1.0, prob_fechar)
    )

    prob_abrir = 1.0 - prob_fechar

    classe_prevista = (
        "FECHAR_PORTA"
        if prob_fechar >= 0.5
        else "ABRIR_PORTA"
    )

    return {
        "prob_abrir": prob_abrir,
        "prob_fechar": prob_fechar,
        "classe_prevista": classe_prevista,
        "tamanho_original": tamanho_original,
        "inicio_fala": inicio_fala,
        "fim_fala": fim_fala,
        "mfcc_shape": mfcc.shape
    }


# ============================================================
# LOCALIZA TODOS OS WAV
# ============================================================

arquivos = sorted(
    AUDIO_DIR.glob("*.wav")
)

if not arquivos:

    raise FileNotFoundError(
        f"Nenhum WAV encontrado em {AUDIO_DIR}"
    )


resultados = []


# ============================================================
# LOOP DOS AUDIOS
# ============================================================

for caminho in arquivos:

    nome = caminho.name.upper()

    # --------------------------------------------------------
    # Determina classe esperada pelo nome
    # --------------------------------------------------------

    if "ABRIR_PORTA" in nome:

        esperado = "ABRIR_PORTA"

    elif "FECHAR_PORTA" in nome:

        esperado = "FECHAR_PORTA"

    else:

        esperado = "DESCONHECIDO"

    resultado = inferir(caminho)

    previsto = resultado["classe_prevista"]

    correto = (
        esperado == previsto
        if esperado != "DESCONHECIDO"
        else False
    )

    # Confianca da classe prevista
    if previsto == "ABRIR_PORTA":

        confianca = resultado["prob_abrir"]

    else:

        confianca = resultado["prob_fechar"]

    resultados.append({
        "arquivo": caminho.name,
        "esperado": esperado,
        "previsto": previsto,
        "prob_abrir": resultado["prob_abrir"],
        "prob_fechar": resultado["prob_fechar"],
        "confianca": confianca,
        "correto": correto
    })


# ============================================================
# RESULTADO INDIVIDUAL
# ============================================================

print("\n")
print("=" * 78)
print("RESULTADOS INDIVIDUAIS")
print("=" * 78)

for r in resultados:

    print()
    print(f"Arquivo           : {r['arquivo']}")
    print(f"Classe esperada   : {r['esperado']}")
    print(
        f"ABRIR_PORTA       : "
        f"{r['prob_abrir'] * 100:.2f} %"
    )
    print(
        f"FECHAR_PORTA      : "
        f"{r['prob_fechar'] * 100:.2f} %"
    )
    print(f"Classe prevista   : {r['previsto']}")
    print(
        f"Resultado         : "
        f"{'CORRETO' if r['correto'] else 'INCORRETO'}"
    )
    print("-" * 78)


# ============================================================
# RESUMO
# ============================================================

print("\n")
print("=" * 78)
print("RESUMO DAS AMOSTRAS INEDITAS")
print("=" * 78)

print(
    f"{'ARQUIVO':32} "
    f"{'ESPERADO':15} "
    f"{'PREVISTO':15} "
    f"{'CONFIANCA':10} "
    f"{'RESULTADO':10}"
)

print("-" * 88)

for r in resultados:

    status = (
        "CORRETO"
        if r["correto"]
        else "INCORRETO"
    )

    print(
        f"{r['arquivo'][:31]:32} "
        f"{r['esperado']:15} "
        f"{r['previsto']:15} "
        f"{r['confianca'] * 100:9.2f}% "
        f"{status:10}"
    )


# ============================================================
# ACURACIA
# ============================================================

validos = [
    r for r in resultados
    if r["esperado"] != "DESCONHECIDO"
]

acertos = sum(
    r["correto"]
    for r in validos
)

total = len(validos)

acuracia = (
    acertos / total
    if total > 0
    else 0
)

print("-" * 88)

print(
    f"Acertos           : {acertos}/{total}"
)

print(
    f"Acuracia          : {acuracia * 100:.2f} %"
)

print("=" * 78)