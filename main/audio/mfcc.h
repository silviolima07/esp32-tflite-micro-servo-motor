#pragma once

#include <stddef.h>
#include <stdint.h>

// Mesmos parâmetros utilizados no treinamento
constexpr int MFCC_SAMPLE_RATE = 16000;
constexpr int MFCC_AUDIO_SAMPLES = 40000;

constexpr int MFCC_N_FFT = 512;
constexpr int MFCC_WIN_LENGTH = 400;
constexpr int MFCC_HOP_LENGTH = 160;
constexpr int MFCC_N_MELS = 40;
constexpr int MFCC_N_COEFFS = 13;

// center=False:
// 1 + floor((40000 - 512) / 160) = 247
constexpr int MFCC_N_FRAMES = 247;

constexpr int MFCC_TOTAL_VALUES =
    MFCC_N_COEFFS * MFCC_N_FRAMES;

/**
 * Calcula os MFCCs a partir da janela PCM já preparada.
 *
 * Entrada:
 *   audio_pcm -> exatamente 40000 amostras PCM16 (2,5 s / 16 kHz)
 *
 * Saída:
 *   mfcc -> matriz linearizada com 13 x 247 valores float
 *
 * Organização:
 *   mfcc[coeficiente * MFCC_N_FRAMES + frame]
 *
 * Exemplo:
 *   MFCC[5,100] =
 *       mfcc[5 * MFCC_N_FRAMES + 100]
 */
bool compute_mfcc(
    const int16_t* audio_pcm,
    size_t num_samples,
    float* mfcc
);