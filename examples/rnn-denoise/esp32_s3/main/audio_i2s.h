#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "driver/i2s_std.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/ringbuf.h"
#include "freertos/task.h"

#define AUDIO_I2S_SAMPLE_RATE 16000
#define AUDIO_I2S_FRAME_SAMPLES 256

typedef struct {
    i2s_chan_handle_t rx_chan;
    i2s_chan_handle_t tx_chan;
    RingbufHandle_t rx_ring;
    TaskHandle_t rx_task;
} audio_i2s_ctx_t;

esp_err_t audio_i2s_init(audio_i2s_ctx_t *ctx);
esp_err_t audio_i2s_start(audio_i2s_ctx_t *ctx);
bool audio_i2s_receive(audio_i2s_ctx_t *ctx, int32_t *dest, size_t frames, TickType_t ticks_to_wait);
esp_err_t audio_i2s_write(audio_i2s_ctx_t *ctx, const int16_t *data, size_t frames, TickType_t ticks_to_wait);
