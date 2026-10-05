#include "audio_preprocessing.h"

#include <math.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// ============================================================
// Parametros equivalentes ao preprocessing Python
// ============================================================

static constexpr size_t FRAME_LENGTH = 2048;
static constexpr size_t HOP_LENGTH   = 512;

static constexpr size_t FRAME_OFFSET = FRAME_LENGTH / 2;

static constexpr float TOP_DB = 30.0f;

static constexpr size_t SAMPLE_RATE = 16000;
static constexpr size_t MARGIN_SAMPLES =
    static_cast<size_t>(0.2f * SAMPLE_RATE);   // 3200


// ============================================================
// Calcula RMS de um frame
// ============================================================

static float calculate_rms(
    const int16_t* audio,
    size_t start,
    size_t total_samples)
{
    double sum = 0.0;

    for (size_t i = 0; i < FRAME_LENGTH; ++i) {

        size_t index = start + i;

        // Padding com zero fora do audio
        float sample = 0.0f;

        if (index < total_samples) {
            sample =
                static_cast<float>(audio[index]) /
                32768.0f;
        }

        sum += static_cast<double>(sample) *
               static_cast<double>(sample);
    }

    return sqrtf(
        static_cast<float>(
            sum / FRAME_LENGTH
        )
    );
}


// ============================================================
// Prepara janela de audio
// ============================================================

bool prepare_audio_window(
    const int16_t* input,
    size_t input_samples,
    int16_t* output,
    size_t output_samples,
    AudioWindowInfo* info)
{
    if (
        input == nullptr ||
        output == nullptr ||
        info == nullptr ||
        input_samples == 0 ||
        output_samples == 0
    ) {
        return false;
    }


    // ========================================================
    // 1. Quantidade de frames
    // ========================================================

    const size_t num_frames =
        (input_samples + HOP_LENGTH - 1)
        / HOP_LENGTH;


    // ========================================================
    // 2. Encontra RMS maximo
    // ========================================================

    float max_rms = 0.0f;

    for (size_t frame = 0;
         frame < num_frames;
         ++frame)
    {
        size_t start =
            frame * HOP_LENGTH;

        float rms = calculate_rms(
            input,
            start,
            input_samples
        );

        if (rms > max_rms) {
            max_rms = rms;
        }
        // Evita watchdog durante simulacao
        if ((frame % 20) == 0) {
            vTaskDelay(1);
        }
    }


    if (max_rms <= 0.0f) {
        return false;
    }


    // ========================================================
    // 3. Threshold equivalente a -30 dB
    //
    // 20 log10(rms/max_rms) >= -30
    //
    // rms >= max_rms * 10^(-30/20)
    // ========================================================

    const float threshold =
        max_rms *
        powf(10.0f, -TOP_DB / 20.0f);


    // ========================================================
    // 4. Localiza primeiro e ultimo frame nao silencioso
    // ========================================================

    size_t first_frame = num_frames;
    size_t last_frame = 0;

    bool found = false;

    for (size_t frame = 0;
         frame < num_frames;
         ++frame)
    {
        size_t start =
            frame * HOP_LENGTH;

        float rms = calculate_rms(
            input,
            start,
            input_samples
        );

        if (rms >= threshold) {

            if (!found) {
                first_frame = frame;
                found = true;
            }

            last_frame = frame;

            // Evita watchdog durante simulacao
            if ((frame % 20) == 0) {
            vTaskDelay(1);
          }
            }
    }


    if (!found) {
        return false;
    }


    // ========================================================
    // 5. Converte frames para indices de amostras
    // ========================================================

    size_t speech_start =
    first_frame * HOP_LENGTH + FRAME_OFFSET;

    size_t speech_end =
    (last_frame + 1) * HOP_LENGTH + FRAME_OFFSET;

    if (speech_start > input_samples) {
    speech_start = input_samples;
    }

    if (speech_end > input_samples) {
    speech_end = input_samples;
    }


    // ========================================================
    // 6. Adiciona margem de 200 ms
    // ========================================================

    size_t margin_start =
        (speech_start > MARGIN_SAMPLES)
        ? speech_start - MARGIN_SAMPLES
        : 0;

    size_t margin_end =
        speech_end + MARGIN_SAMPLES;

    if (margin_end > input_samples) {
        margin_end = input_samples;
    }


    size_t cropped_samples =
        margin_end - margin_start;


    // ========================================================
    // 7. Inicializa output com zeros
    // ========================================================

    memset(
        output,
        0,
        output_samples * sizeof(int16_t)
    );


    size_t padding_before = 0;
    size_t padding_after = 0;


    // ========================================================
    // 8. Recorte menor que 40000 -> padding centralizado
    // ========================================================

    if (cropped_samples < output_samples) {

        size_t missing =
            output_samples - cropped_samples;

        padding_before =
            missing / 2;

        padding_after =
            missing - padding_before;

        memcpy(
            output + padding_before,
            input + margin_start,
            cropped_samples * sizeof(int16_t)
        );
    }

    // ========================================================
    // 9. Recorte maior que 40000 -> crop centralizado
    // ========================================================

    else if (cropped_samples > output_samples) {

        size_t excess =
            cropped_samples - output_samples;

        size_t crop_start =
            margin_start + excess / 2;

        memcpy(
            output,
            input + crop_start,
            output_samples * sizeof(int16_t)
        );

        padding_before = 0;
        padding_after = 0;
    }

    // ========================================================
    // 10. Exatamente 40000
    // ========================================================

    else {

        memcpy(
            output,
            input + margin_start,
            output_samples * sizeof(int16_t)
        );
    }


    // ========================================================
    // Informacoes para diagnostico
    // ========================================================

    info->speech_start = speech_start;
    info->speech_end = speech_end;

    info->margin_start = margin_start;
    info->margin_end = margin_end;

    info->cropped_samples = cropped_samples;

    info->padding_before = padding_before;
    info->padding_after = padding_after;


    return true;
}