#pragma once

#include <stddef.h>
#include <stdint.h>

struct AudioWindowInfo {
    size_t speech_start;
    size_t speech_end;

    size_t margin_start;
    size_t margin_end;

    size_t cropped_samples;

    size_t padding_before;
    size_t padding_after;
};

bool prepare_audio_window(
    const int16_t* input,
    size_t input_samples,
    int16_t* output,
    size_t output_samples,
    AudioWindowInfo* info
);