#include <stdio.h>
#include <stdint.h>
#include "driver/usb_serial_jtag.h"

#include <math.h>
#include <string.h>

// FreeRTOS
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Modelo
#include "model.h"

// Audio
#include "teste_abrir_01.h"
#include "audio_preprocessing.h"

// Servo Motor
#include "servo_motor.h"

// Audios de teste
#include "teste_abrir_01.h"
#include "teste_abrir_02.h"
#include "teste_fechar_01.h"
#include "teste_fechar_02.h"

#include "teste_05.h"
#include "teste_06.h"

#include "audio/validacao_fechar_01.h"

#include "mfcc.h"

#include "esp_heap_caps.h"

static constexpr size_t AUDIO_FINAL_SAMPLES = 40000;
// static int16_t audio_processado[AUDIO_FINAL_SAMPLES];

constexpr int AUDIO_SAMPLE_RATE = 16000;

// Memória usada pelo TensorFlow Lite Micro para os tensores.
// Podemos ajustar depois, se necessário.
constexpr int kTensorArenaSize = 200 * 1024;

alignas(16) static uint8_t tensor_arena[kTensorArenaSize];


struct AudioTeste {
    const int16_t* pcm;
    int num_samples;
    const char* arquivo_real;
    const char* classe_real;
};

static const AudioTeste audios_teste[] = {
    {
        teste_abrir_01_pcm,
        teste_abrir_01_num_samples,
        "Teste_ABRIR_PORTA_01.wav",
        "ABRIR_PORTA"
    },
    {
        teste_abrir_02_pcm,
        teste_abrir_02_num_samples,
        "Teste_ABRIR_PORTA_02.wav",
        "ABRIR_PORTA"
    },
    {
        teste_fechar_01_pcm,
        teste_fechar_01_num_samples,
        "Teste_FECHAR_PORTA_01.wav",
        "FECHAR_PORTA"
    },
    {
        teste_fechar_02_pcm,
        teste_fechar_02_num_samples,
        "Teste_FECHAR_PORTA_02.wav",
        "FECHAR_PORTA"
    }
    ,
    {
        teste_05_pcm,
        teste_05_num_samples,
        "matheus_eleven_fechar_porta.wav",
        "FECHAR_PORTA"
    },
    {
        teste_06_pcm,
        teste_06_num_samples,
        "igor_audacity_abrir_porta.wav",
        "ABRIR_PORTA"
    }
    
};

extern "C" void app_main(void)
{
    printf("\n");
    printf("=====================================\n");
    printf(" TinyML - Comandos de Voz\n");
    printf(" ESP32-S3 + TensorFlow Lite Micro\n");
    printf("=====================================\n");

    printf("Tamanho do modelo: %u bytes\n", g_model_len);

    // ---------------------------------------------------------
    // 1. Carrega o modelo
    // ---------------------------------------------------------

    const tflite::Model* model = tflite::GetModel(g_model);

    if (model == nullptr)
    {
        printf("ERRO: nao foi possivel carregar o modelo.\n");
        return;
    }

    printf("Modelo carregado com sucesso.\n");
    printf("Schema do modelo: %lu\n",
           static_cast<unsigned long>(model->version()));
    printf("Schema esperado: %d\n", TFLITE_SCHEMA_VERSION);

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        printf("ERRO: schema incompativel.\n");
        return;
    }

    printf("Schema compativel.\n");

    // ---------------------------------------------------------
    // 2. Operacoes utilizadas pela nossa CNN
    // ---------------------------------------------------------

    static tflite::MicroMutableOpResolver<8> resolver;

    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddLogistic();
    resolver.AddReshape();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();

    // ---------------------------------------------------------
    // 3. Cria o interpretador
    // ---------------------------------------------------------

    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    // ---------------------------------------------------------
    // 4. Reserva memoria para os tensores
    // ---------------------------------------------------------

    TfLiteStatus status = interpreter.AllocateTensors();

    if (status != kTfLiteOk)
    {
        printf("ERRO: AllocateTensors() falhou.\n");
        return;
    }

    printf("AllocateTensors(): OK\n");

    // ---------------------------------------------------------
    // 5. Entrada e saida do modelo
    // ---------------------------------------------------------

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("\n--- INPUT ---\n");

    printf("Tipo: %d\n", input->type);

    printf("Shape: ");
    for (int i = 0; i < input->dims->size; i++)
    {
        printf("%d", input->dims->data[i]);

        if (i < input->dims->size - 1)
            printf(" x ");
    }

    printf("\n");

    printf("Scale: %f\n", input->params.scale);
    printf("Zero point: %ld\n",
           static_cast<long>(input->params.zero_point));

    printf("\n--- OUTPUT ---\n");

    printf("Tipo: %d\n", output->type);

    printf("Shape: ");
    for (int i = 0; i < output->dims->size; i++)
    {
        printf("%d", output->dims->data[i]);

        if (i < output->dims->size - 1)
            printf(" x ");
    }

    printf("\n");

    printf("Scale: %f\n", output->params.scale);
    printf("Zero point: %ld\n",
           static_cast<long>(output->params.zero_point));
    
    printf("\n");
printf("========================================\n");
printf("       SELECAO DO AUDIO DE TESTE\n");
printf("========================================\n\n");

printf("1 - Audio de teste 01\n");
printf("2 - Audio de teste 02\n");
printf("3 - Audio de teste 03\n");
printf("4 - Audio de teste 04\n\n");
printf("5 - Pessoa A - Audio de teste 05\n");
printf("6 - Pessoa B - Audio de teste 06\n\n");


usb_serial_jtag_driver_config_t usb_config = {
    .tx_buffer_size = 256,
    .rx_buffer_size = 256,
};

ESP_ERROR_CHECK(
    usb_serial_jtag_driver_install(&usb_config)
);

printf("Escolha um audio: ");
fflush(stdout);

int opcao = 0;
uint8_t tecla = 0;

while (opcao < 1 || opcao > 6) {

    int len = usb_serial_jtag_read_bytes(
        &tecla,
        1,
        pdMS_TO_TICKS(100)
    );

    if (len > 0 && tecla >= '1' && tecla <= '6') {
        opcao = tecla - '0';
        printf("%d\n", opcao);
    }
}


// Audio escolhido pelo usuario
const AudioTeste& audio = audios_teste[opcao - 1];

printf("\nAudio selecionado : %02d\n", opcao);
printf("Numero de amostras: %d\n", audio.num_samples);
printf("Sample rate       : %d Hz\n", AUDIO_SAMPLE_RATE);
printf("Duracao original  : %.3f s\n",
       (float)audio.num_samples / (float)AUDIO_SAMPLE_RATE);


// --------------------------------------------------
// PRE-PROCESSAMENTO DO AUDIO SELECIONADO
// --------------------------------------------------

AudioWindowInfo audio_info;

int16_t* audio_processado =
    static_cast<int16_t*>(
        heap_caps_malloc(
            MFCC_AUDIO_SAMPLES * sizeof(int16_t),
            MALLOC_CAP_8BIT
        )
    );

if (audio_processado == nullptr) {
    printf("ERRO: nao foi possivel alocar buffer de audio\n");
    return;
}

bool audio_ok = prepare_audio_window(
    audio.pcm,
    audio.num_samples,
    audio_processado,
    40000,
    &audio_info
);

// --------------------------------------------------
// TESTE DO SERVO MOTOR
// --------------------------------------------------

servo_init();

printf("\nServo inicializado.\n");
printf("Aguardando resultado da inferencia...\n");     



printf("\nPRE-PROCESSAMENTO DO AUDIO\n");
printf("--------------------------------------------------\n");

if (!audio_ok) {

    printf("Resultado           : ERRO\n");
    heap_caps_free(audio_processado);
    return;


} else {

    printf("Amostras originais  : %lu\n",
           (unsigned long)audio.num_samples);

    printf("\nFALA DETECTADA\n");
    printf("Inicio              : %lu\n",
           (unsigned long)audio_info.speech_start);
    printf("Fim                 : %lu\n",
           (unsigned long)audio_info.speech_end);

    printf("\nMARGEM DE 200 ms\n");
    printf("Inicio              : %lu\n",
           (unsigned long)audio_info.margin_start);
    printf("Fim                 : %lu\n",
           (unsigned long)audio_info.margin_end);

    printf("\nJANELA FINAL\n");
    printf("Recorte             : %lu\n",
           (unsigned long)audio_info.cropped_samples);
    printf("Padding antes       : %lu\n",
           (unsigned long)audio_info.padding_before);
    printf("Padding depois      : %lu\n",
           (unsigned long)audio_info.padding_after);
    printf("Amostras finais     : 40000\n");
    printf("Duracao final       : 2.500 s\n");
    printf("Resultado           : OK\n");

    printf("\nVALIDACAO DO BUFFER PCM\n");
    printf("--------------------------------------------------\n");

    // Primeiro sample real dentro da janela de 40000
    size_t primeiro_processado = audio_info.padding_before;

    // Último sample real dentro da janela
    size_t ultimo_processado =
        audio_info.padding_before + audio_info.cropped_samples - 1;

    // Dois pontos internos para comparação
    size_t ponto1_processado =
        audio_info.padding_before + audio_info.cropped_samples / 3;

    size_t ponto2_processado =
        audio_info.padding_before + (2 * audio_info.cropped_samples) / 3;

    // Posições equivalentes no PCM original
    size_t ponto1_original =
        audio_info.margin_start + audio_info.cropped_samples / 3;

    size_t ponto2_original =
        audio_info.margin_start + (2 * audio_info.cropped_samples) / 3;


    printf("Inicio da janela\n");
    printf("audio_processado[0] : %d\n",
        audio_processado[0]);

    printf("Primeira amostra real\n");
    printf("processado[%lu] : %d\n",
        (unsigned long)primeiro_processado,
        audio_processado[primeiro_processado]);

    printf("original[%lu]   : %d\n",
        (unsigned long)audio_info.margin_start,
        audio.pcm[audio_info.margin_start]);


    printf("\nComparacao no meio do audio\n");
    printf("--------------------------------------------------\n");

    printf("processado[%lu] : %d\n",
        (unsigned long)ponto1_processado,
        audio_processado[ponto1_processado]);

    printf("original[%lu]   : %d\n",
        (unsigned long)ponto1_original,
        audio.pcm[ponto1_original]);

    printf("\n");

    printf("processado[%lu] : %d\n",
        (unsigned long)ponto2_processado,
        audio_processado[ponto2_processado]);

    printf("original[%lu]   : %d\n",
        (unsigned long)ponto2_original,
        audio.pcm[ponto2_original]);


    printf("\nFinal do trecho\n");
    printf("--------------------------------------------------\n");

    printf("Ultima amostra real\n");
    printf("processado[%lu] : %d\n",
        (unsigned long)ultimo_processado,
        audio_processado[ultimo_processado]);

    printf("original[%lu]   : %d\n",
        (unsigned long)(audio_info.margin_end - 1),
        audio.pcm[audio_info.margin_end - 1]);

    printf("audio_processado[39999] : %d\n",
        audio_processado[39999]);
    

}

// ============================================================
// CALCULO DO MFCC
// ============================================================

static float mfcc[MFCC_TOTAL_VALUES];

printf("\nCALCULO DO MFCC\n");
printf("--------------------------------------------------\n");

bool mfcc_ok = compute_mfcc(
    audio_processado,
    MFCC_AUDIO_SAMPLES,
    mfcc
);

if (!mfcc_ok) {
    printf("ERRO: falha no calculo do MFCC\n");
    return;
}

printf("Resultado           : OK\n");
printf("Coeficientes        : %d\n", MFCC_N_COEFFS);
printf("Frames              : %d\n", MFCC_N_FRAMES);
printf("Total de valores    : %d\n", MFCC_TOTAL_VALUES);

printf("\nAMOSTRA DOS MFCCs\n");
printf("--------------------------------------------------\n");

for (int c = 0; c < MFCC_N_COEFFS; ++c) {
    printf(
        "MFCC[%02d][0] = %.6f\n",
        c,
        mfcc[c * MFCC_N_FRAMES]
    );
}

// Permite que o scheduler execute antes da inferencia
vTaskDelay(pdMS_TO_TICKS(10));


// ============================================================
// VALIDACAO MFCC - AUDIO USADO NO TREINAMENTO
// FECHAR_PORTA_1.wav
// ============================================================

/* printf("\n");
printf("==================================================\n");
printf(" VALIDACAO MFCC - AUDIO DE TREINAMENTO\n");
printf("==================================================\n");

printf("Arquivo             : FECHAR_PORTA_1.wav\n");
printf("Amostras originais  : %lu\n",
       (unsigned long)validacao_fechar_01_num_samples);

AudioWindowInfo validacao_info;



bool validacao_audio_ok = prepare_audio_window(
    validacao_fechar_01_pcm,
    validacao_fechar_01_num_samples,
    audio_processado,
    MFCC_AUDIO_SAMPLES,
    &validacao_info
);

if (!validacao_audio_ok) {
    printf("ERRO: falha no pre-processamento do audio de validacao\n");
    return;
}

printf("\nPRE-PROCESSAMENTO\n");
printf("Fala inicio         : %lu\n",
       (unsigned long)validacao_info.speech_start);
printf("Fala fim            : %lu\n",
       (unsigned long)validacao_info.speech_end);
printf("Margem inicio       : %lu\n",
       (unsigned long)validacao_info.margin_start);
printf("Margem fim          : %lu\n",
       (unsigned long)validacao_info.margin_end);
printf("Recorte             : %lu\n",
       (unsigned long)validacao_info.cropped_samples);
printf("Padding antes       : %lu\n",
       (unsigned long)validacao_info.padding_before);
printf("Padding depois      : %lu\n",
       (unsigned long)validacao_info.padding_after);

static float mfcc_validacao[MFCC_TOTAL_VALUES];

printf("\nCALCULANDO MFCC DE VALIDACAO...\n");

bool validacao_mfcc_ok = compute_mfcc(
    audio_processado,
    MFCC_AUDIO_SAMPLES,
    mfcc_validacao
);

if (!validacao_mfcc_ok) {
    printf("ERRO: falha no MFCC de validacao\n");
    return;
}

printf("Resultado            : OK\n");
printf("Dimensao             : %d x %d\n",
       MFCC_N_COEFFS,
       MFCC_N_FRAMES);

printf("\nVALORES PARA COMPARACAO COM PYTHON\n");
printf("--------------------------------------------------\n");

const int frames_validacao[] = {
    0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 246
};

for (int c = 0; c < MFCC_N_COEFFS; ++c) {

    printf("\nMFCC[%02d]\n", c);

    for (int i = 0;
         i < static_cast<int>(
             sizeof(frames_validacao) /
             sizeof(frames_validacao[0])
         );
         ++i) {

        int frame = frames_validacao[i];

        printf(
            "  frame %03d = %.6f\n",
            frame,
            mfcc_validacao[c * MFCC_N_FRAMES + frame]
        );
    }
}

printf("\nVALIDACAO MFCC ESP32 CONCLUIDA\n");
printf("==================================================\n"); */

// ---------------------------------------------------------
// 6. Teste controlado de inferencia
// ---------------------------------------------------------

printf("\n");
printf("==================================================\n");
printf("       TINYML - RECONHECIMENTO DE COMANDO\n");
printf("==================================================\n");

printf("\nENTRADA\n");
printf("--------------------------------------------------\n");
printf("Audio selecionado  : %02d\n", opcao);
printf("MFCC               : %d x %d\n",
       MFCC_N_COEFFS,
       MFCC_N_FRAMES);
printf("Tensor             : 1 x 13 x 247 x 1\n");
printf("Tipo               : INT8\n");

// ------------------------------------------------------------
// Normalizacao utilizada durante o treinamento
// ------------------------------------------------------------

constexpr float MFCC_MEDIA  = -28.254616f;
constexpr float MFCC_DESVIO = 117.86187f;

printf("\nNORMALIZACAO E QUANTIZACAO\n");
printf("--------------------------------------------------\n");
printf("Media treinamento  : %.6f\n", MFCC_MEDIA);
printf("Desvio treinamento : %.6f\n", MFCC_DESVIO);
printf("Input scale        : %.8f\n", input->params.scale);
printf("Input zero point   : %ld\n",
       static_cast<long>(input->params.zero_point));

// ------------------------------------------------------------
// MFCC -> normalizacao -> INT8
// ------------------------------------------------------------

const int total_elementos =
    MFCC_N_COEFFS * MFCC_N_FRAMES;

for (int i = 0; i < total_elementos; ++i)
{
    // Mesma normalizacao utilizada no treinamento
    float normalizado =
        (mfcc[i] - MFCC_MEDIA) / MFCC_DESVIO;

    // Quantizacao para o tensor INT8:
    //
    // q = round(real / scale) + zero_point
    //
    int32_t quantizado =
        static_cast<int32_t>(
            roundf(
                normalizado /
                input->params.scale
            )
        ) +
        input->params.zero_point;

    // Saturacao INT8
    if (quantizado > 127)
        quantizado = 127;

    if (quantizado < -128)
        quantizado = -128;

    input->data.int8[i] =
        static_cast<int8_t>(quantizado);
}

printf("Entrada preparada  : OK\n");

// ------------------------------------------------------------
// Inferencia
// ------------------------------------------------------------

printf("\nINFERENCIA\n");
printf("--------------------------------------------------\n");

TfLiteStatus invoke_status =
    interpreter.Invoke();

if (invoke_status != kTfLiteOk)
{
    printf("ERRO: Invoke() falhou.\n");
    heap_caps_free(audio_processado);
    return;
}

// ------------------------------------------------------------
// Desquantizacao da saida
// ------------------------------------------------------------

int8_t output_int8 =
    output->data.int8[0];

float prob_fechar =
    (
        static_cast<int32_t>(output_int8) -
        output->params.zero_point
    ) *
    output->params.scale;

// Protecao numerica
if (prob_fechar < 0.0f)
    prob_fechar = 0.0f;

if (prob_fechar > 1.0f)
    prob_fechar = 1.0f;

float prob_abrir =
    1.0f - prob_fechar;

printf("ABRIR_PORTA       : %.2f %%\n",
       prob_abrir * 100.0f);

printf("FECHAR_PORTA      : %.2f %%\n",
       prob_fechar * 100.0f);

// ------------------------------------------------------------
// Classe prevista
// ------------------------------------------------------------

const char* classe_prevista;

if (prob_fechar >= 0.5f)
{
    classe_prevista = "FECHAR_PORTA";
}
else
{
    classe_prevista = "ABRIR_PORTA";
}

printf("\nClasse prevista   : %s\n",
       classe_prevista);

// ------------------------------------------------------------
// Aciona o servo SOMENTE depois da classificacao
// ------------------------------------------------------------

printf("\nACAO\n");
printf("--------------------------------------------------\n");

if (prob_fechar >= 0.5f)
{
    printf("Comando reconhecido: FECHAR PORTA\n");
    fechar_porta();
}
else
{
    printf("Comando reconhecido: ABRIR PORTA\n");
    abrir_porta();
}

vTaskDelay(pdMS_TO_TICKS(1000));

// ------------------------------------------------------------
// Somente agora revelamos a resposta correta
// ------------------------------------------------------------

printf("\nVALIDACAO CEGA\n");
printf("--------------------------------------------------\n");

printf("Arquivo real      : %s\n",
       audio.arquivo_real);

printf("Classe real       : %s\n",
       audio.classe_real);

printf("Classe prevista   : %s\n",
       classe_prevista);

bool acertou =
    strcmp(
        classe_prevista,
        audio.classe_real
    ) == 0;

printf("Resultado         : %s\n",
       acertou ? "CORRETO" : "INCORRETO");

printf("\n==================================================\n");

if (acertou)
{
    printf("              TESTE CORRETO\n");
}
else
{
    printf("             TESTE INCORRETO\n");
}

printf("==================================================\n");

// ------------------------------------------------------------
// Libera memoria
// ------------------------------------------------------------

heap_caps_free(audio_processado);

printf("\nTeste concluido.\n");
}