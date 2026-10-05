#include "mfcc.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <cmath>
#include <cstring>

#include "esp_dsp.h"

static constexpr float PI_F = 3.14159265358979323846f;
static constexpr int FFT_BINS = MFCC_N_FFT / 2 + 1;

static float fft_buffer[MFCC_N_FFT * 2];
static float power_spectrum[FFT_BINS];
static float hann_window[MFCC_WIN_LENGTH];

static float mel_left[MFCC_N_MELS];
static float mel_center[MFCC_N_MELS];
static float mel_right[MFCC_N_MELS];
static float mel_norm[MFCC_N_MELS];

static bool mfcc_initialized = false;


// ------------------------------------------------------------
// Hz <-> Mel - escala Slaney
// ------------------------------------------------------------

static float hz_to_mel(float hz)
{
    constexpr float f_sp = 200.0f / 3.0f;

    float mel = hz / f_sp;

    if (hz >= 1000.0f) {
        constexpr float min_log_hz = 1000.0f;
        constexpr float min_log_mel = min_log_hz / f_sp;

        const float logstep =
            std::log(6.4f) / 27.0f;

        mel =
            min_log_mel +
            std::log(hz / min_log_hz) / logstep;
    }

    return mel;
}


static float mel_to_hz(float mel)
{
    constexpr float f_sp = 200.0f / 3.0f;
    constexpr float min_log_hz = 1000.0f;
    constexpr float min_log_mel = min_log_hz / f_sp;

    float hz = mel * f_sp;

    if (mel >= min_log_mel) {

        const float logstep =
            std::log(6.4f) / 27.0f;

        hz =
            min_log_hz *
            std::exp(
                logstep *
                (mel - min_log_mel)
            );
    }

    return hz;
}


// ------------------------------------------------------------
// Janela Hann periódica
// ------------------------------------------------------------

static void create_hann_window()
{
    for (int n = 0; n < MFCC_WIN_LENGTH; ++n) {

        hann_window[n] =
            0.5f -
            0.5f *
            std::cos(
                2.0f *
                PI_F *
                n /
                MFCC_WIN_LENGTH
            );
    }
}


// ------------------------------------------------------------
// Apenas limites dos filtros Mel.
//
// Não armazenamos mais:
// float mel_filters[40][257]
// ------------------------------------------------------------

static void create_mel_filters()
{
    float mel_points[MFCC_N_MELS + 2];
    float hz_points[MFCC_N_MELS + 2];

    const float mel_min =
        hz_to_mel(0.0f);

    const float mel_max =
        hz_to_mel(
            MFCC_SAMPLE_RATE / 2.0f
        );

    for (int i = 0; i < MFCC_N_MELS + 2; ++i) {

        const float alpha =
            static_cast<float>(i) /
            static_cast<float>(MFCC_N_MELS + 1);

        mel_points[i] =
            mel_min +
            alpha *
            (mel_max - mel_min);

        hz_points[i] =
            mel_to_hz(mel_points[i]);
    }

    for (int m = 0; m < MFCC_N_MELS; ++m) {

        mel_left[m] =
            hz_points[m];

        mel_center[m] =
            hz_points[m + 1];

        mel_right[m] =
            hz_points[m + 2];

        mel_norm[m] =
            2.0f /
            (mel_right[m] - mel_left[m]);
    }
}


// ------------------------------------------------------------
// Inicialização
// ------------------------------------------------------------

static bool initialize_mfcc()
{
    if (mfcc_initialized) {
        return true;
    }

    create_hann_window();
    create_mel_filters();

    esp_err_t ret =
        dsps_fft2r_init_fc32(
            nullptr,
            MFCC_N_FFT
        );

    if (ret != ESP_OK) {
        return false;
    }

    mfcc_initialized = true;

    return true;
}


// ------------------------------------------------------------
// Peso de um filtro Mel calculado sob demanda
// ------------------------------------------------------------

static float mel_weight(
    int mel_index,
    float frequency
)
{
    const float left =
        mel_left[mel_index];

    const float center =
        mel_center[mel_index];

    const float right =
        mel_right[mel_index];

    float weight = 0.0f;

    if (
        frequency >= left &&
        frequency <= center
    ) {

        weight =
            (frequency - left) /
            (center - left);

    } else if (
        frequency > center &&
        frequency <= right
    ) {

        weight =
            (right - frequency) /
            (right - center);
    }

    return
        weight *
        mel_norm[mel_index];
}


// ------------------------------------------------------------
// Calcula o espectro de potência de um frame
// ------------------------------------------------------------

static void calculate_power_spectrum(
    const int16_t* audio_pcm,
    int frame
)
{
    memset(
        fft_buffer,
        0,
        sizeof(fft_buffer)
    );

    constexpr int WINDOW_OFFSET =
        (MFCC_N_FFT - MFCC_WIN_LENGTH) / 2;

    const int start =
        frame * MFCC_HOP_LENGTH;

    for (int n = 0; n < MFCC_WIN_LENGTH; ++n) {

        const int audio_index =
            start +
            WINDOW_OFFSET +
            n;

        const float sample =
            static_cast<float>(
                audio_pcm[audio_index]
            ) / 32768.0f;

        fft_buffer[
            2 * (WINDOW_OFFSET + n)
        ] =
            sample *
            hann_window[n];

        fft_buffer[
            2 * (WINDOW_OFFSET + n) + 1
        ] = 0.0f;
    }
    
    // FFT usando explicitamente a implementação ANSI.
    // Esta versão foi validada contra o NumPy/librosa.

    dsps_fft2r_fc32_ansi(
        fft_buffer,
        MFCC_N_FFT
    );

    dsps_bit_rev_fc32_ansi(
        fft_buffer,
        MFCC_N_FFT
    );

    // Calcula o espectro de potência
    for (int k = 0; k < FFT_BINS; ++k) {

        const float real =
            fft_buffer[2 * k];

        const float imag =
            fft_buffer[2 * k + 1];

        power_spectrum[k] =
            real * real +
            imag * imag;
    }

    }

// ------------------------------------------------------------
// Energia de um filtro Mel
// ------------------------------------------------------------

static float calculate_mel_energy(
    int mel_index
)
{
    float energy = 0.0f;

    for (int k = 0; k < FFT_BINS; ++k) {

        const float frequency =
            static_cast<float>(k) *
            MFCC_SAMPLE_RATE /
            MFCC_N_FFT;

        const float weight =
            mel_weight(
                mel_index,
                frequency
            );

        energy +=
            power_spectrum[k] *
            weight;
    }

    if (energy < 1.0e-10f) {
        energy = 1.0e-10f;
    }

    return energy;
}


// ------------------------------------------------------------
// MFCC
//
// Duas passagens:
//
// 1. encontra máximo global do Mel spectrogram
// 2. recalcula e gera dB + DCT
//
// Gastamos processamento para economizar DRAM.
// ------------------------------------------------------------

bool compute_mfcc(
    const int16_t* audio_pcm,
    size_t num_samples,
    float* mfcc
)
{
    if (
        audio_pcm == nullptr ||
        mfcc == nullptr ||
        num_samples != MFCC_AUDIO_SAMPLES
    ) {
        return false;
    }

    if (!initialize_mfcc()) {
        return false;
    }


    // ========================================================
    // PASSAGEM 1
    //
    // Encontra máximo global necessário para top_db=80
    // ========================================================

    float global_max = 1.0e-10f;

    for (int frame = 0;
         frame < MFCC_N_FRAMES;
         ++frame)
    {
        calculate_power_spectrum(
            audio_pcm,
            frame
        );

        for (int m = 0;
             m < MFCC_N_MELS;
             ++m) {

            const float energy =
                calculate_mel_energy(m);

            if (energy > global_max) {
                global_max = energy;
            }
        }

        // Libera a CPU periodicamente para o FreeRTOS
        if ((frame % 5) == 0) {
            vTaskDelay(1);
        }
    }


    const float max_db =
        10.0f *
        std::log10(global_max);

    const float min_db =
        max_db - 80.0f;


    // ========================================================
    // PASSAGEM 2
    //
    // Recalcula cada frame e produz diretamente seus
    // 13 coeficientes MFCC.
    // ========================================================

    float mel_db[MFCC_N_MELS];

    for (int frame = 0;
         frame < MFCC_N_FRAMES;
         ++frame)
    {
        calculate_power_spectrum(
            audio_pcm,
            frame
        );


        // Mel -> dB
        for (int m = 0;
             m < MFCC_N_MELS;
             ++m) {

            const float energy =
                calculate_mel_energy(m);

            float db =
                10.0f *
                std::log10(energy);

            if (db < min_db) {
                db = min_db;
            }

            mel_db[m] = db;
        }


        // DCT-II ortonormal
        for (int c = 0;
             c < MFCC_N_COEFFS;
             ++c) {

            float sum = 0.0f;

            for (int m = 0;
                 m < MFCC_N_MELS;
                 ++m) {

                const float angle =
                    PI_F *
                    c *
                    (m + 0.5f) /
                    MFCC_N_MELS;

                sum +=
                    mel_db[m] *
                    std::cos(angle);
            }

            const float scale =
                (c == 0)
                ? std::sqrt(
                    1.0f /
                    MFCC_N_MELS
                  )
                : std::sqrt(
                    2.0f /
                    MFCC_N_MELS
                  );

            mfcc[
                c * MFCC_N_FRAMES +
                frame
            ] =
                sum * scale;
        }

        // Libera a CPU periodicamente para o FreeRTOS
        if ((frame % 5) == 0) {
            vTaskDelay(1);
        }
    }

    return true;
}