#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_i2s.h"
#include "denoise_weights.h"
#include "equalizer_coeff.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "mfcc.h"
#include "nnom.h"

#define NUM_FEATURES NUM_FILTER

#define _MAX(x, y) (((x) > (y)) ? (x) : (y))
#define _MIN(x, y) (((x) < (y)) ? (x) : (y))

#define NUM_CHANNELS 1
#define SAMPLE_RATE 16000
#define AUDIO_FRAME_LEN 512
#define AUDIO_HOP_LEN (AUDIO_FRAME_LEN / 2)

#ifndef NNOM_DEBUG_FIXED_FRAME
#define NNOM_DEBUG_FIXED_FRAME 1
#endif

#define SaturaLH(N, L, H) (((N) < (L)) ? (L) : (((N) > (H)) ? (H) : (N)))

static const char *TAG = "rnn_denoise";

static const int16_t kDebugFrame[AUDIO_HOP_LEN] = {
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827,
    0, 3827, 7071, 9239, 10000, 9239, 7071, 3827, 0, -3827, -7071, -9239, -10000, -9239, -7071, -3827
};

// audio buffer for input
static float audio_buffer[AUDIO_FRAME_LEN] = {0};
static int16_t audio_buffer_16bit[AUDIO_FRAME_LEN] = {0};

// buffer for output
static int16_t audio_buffer_filtered[AUDIO_HOP_LEN] = {0};

// mfcc features and their derivatives
static float mfcc_feature[NUM_FEATURES] = {0};
static float mfcc_feature_prev[NUM_FEATURES] = {0};
static float mfcc_feature_diff[NUM_FEATURES] = {0};
static float mfcc_feature_diff_prev[NUM_FEATURES] = {0};
static float mfcc_feature_diff1[NUM_FEATURES] = {0};
// features for NN
static float nn_features[64] = {0};
static int8_t nn_features_q7[64] = {0};

// NN results, which is the gains for each frequency band
static float band_gains[NUM_FILTER] = {0};
static float band_gains_prev[NUM_FILTER] = {0};

// 0db gains coefficient
static float coeff_b[NUM_FILTER][NUM_COEFF_PAIR] = FILTER_COEFF_B;
static float coeff_a[NUM_FILTER][NUM_COEFF_PAIR] = FILTER_COEFF_A;
// dynamic gains coefficient
static float b_[NUM_FILTER][NUM_COEFF_PAIR] = {0};

static audio_i2s_ctx_t audio_ctx;
static nnom_model_t *model;

// update the history
static void y_h_update(float *y_h, uint32_t len)
{
    for (uint32_t i = len - 1; i > 0; i--) {
        y_h[i] = y_h[i - 1];
    }
}

//  equalizer by multiple n order iir band pass filter.
// y[i] = b[0] * x[i] + b[1] * x[i - 1] + b[2] * x[i - 2] - a[1] * y[i - 1] - a[2] * y[i - 2]...
static void equalizer(float *x, float *y, uint32_t signal_len, float *b, float *a, uint32_t num_band, uint32_t num_order)
{
    // the y history for each band
    static float y_h[NUM_FILTER][NUM_COEFF_PAIR] = {0};
    static float x_h[NUM_COEFF_PAIR * 2] = {0};
    uint32_t num_coeff = num_order * 2 + 1;

    // i <= num_coeff (where historical x is involved in the first few points)
    // combine state and new data to get a continual x input.
    memcpy(x_h + num_coeff, x, num_coeff * sizeof(float));
    for (uint32_t i = 0; i < num_coeff; i++) {
        y[i] = 0;
        for (uint32_t n = 0; n < num_band; n++) {
            y_h_update(y_h[n], num_coeff);
            y_h[n][0] = b[n * num_coeff] * x_h[i + num_coeff];
            for (uint32_t c = 1; c < num_coeff; c++) {
                y_h[n][0] += b[n * num_coeff + c] * x_h[num_coeff + i - c] - a[n * num_coeff + c] * y_h[n][c];
            }
            y[i] += y_h[n][0];
        }
    }
    // store the x for the state of next round
    memcpy(x_h, &x[signal_len - num_coeff], num_coeff * sizeof(float));

    // i > num_coeff; the rest data not involed the x history
    for (uint32_t i = num_coeff; i < signal_len; i++) {
        y[i] = 0;
        for (uint32_t n = 0; n < num_band; n++) {
            y_h_update(y_h[n], num_coeff);
            y_h[n][0] = b[n * num_coeff] * x[i];
            for (uint32_t c = 1; c < num_coeff; c++) {
                y_h[n][0] += b[n * num_coeff + c] * x[i - c] - a[n * num_coeff + c] * y_h[n][c];
            }
            y[i] += y_h[n][0];
        }
    }
}

// set dynamic gains. Multiple gains x b_coeff
static void set_gains(float *b_in, float *b_out, float *gains, uint32_t num_band, uint32_t num_order)
{
    uint32_t num_coeff = num_order * 2 + 1;
    for (uint32_t i = 0; i < num_band; i++) {
        for (uint32_t c = 0; c < num_coeff; c++) {
            b_out[num_coeff * i + c] = b_in[num_coeff * i + c] * gains[i];
        }
    }
}

static void quantize_data(float *din, int8_t *dout, uint32_t size, uint32_t int_bit)
{
    float limit = (1 << int_bit);
    for (uint32_t i = 0; i < size; i++) {
        dout[i] = (int8_t)(_MAX(_MIN(din[i], limit), -limit) / limit * 127);
    }
}

static void print_vector(const char *label, const float *values, uint32_t size)
{
    printf("%s", label);
    for (uint32_t i = 0; i < size; i++) {
        printf("%f%s", values[i], (i + 1 == size) ? "\n" : ",");
    }
}

static void debug_fixed_frame(mfcc_t *mfcc)
{
#if NNOM_DEBUG_FIXED_FRAME
    float debug_mfcc_feature[NUM_FEATURES] = {0};
    float debug_mfcc_feature_prev[NUM_FEATURES] = {0};
    float debug_mfcc_feature_diff[NUM_FEATURES] = {0};
    float debug_mfcc_feature_diff_prev[NUM_FEATURES] = {0};
    float debug_mfcc_feature_diff1[NUM_FEATURES] = {0};
    float debug_nn_features[64] = {0};
    int8_t debug_nn_features_q7[64] = {0};
    float debug_band_gains[NUM_FEATURES] = {0};
    float debug_band_gains_prev[NUM_FEATURES] = {0};

    nnom_model_t *debug_model = nnom_model_create();
    if (debug_model == NULL) {
        ESP_LOGE(TAG, "failed to create debug model");
        return;
    }

    mfcc_compute(mfcc, kDebugFrame, debug_mfcc_feature);

    for (uint32_t i = 0; i < NUM_FEATURES; i++) {
        debug_mfcc_feature_diff[i] = debug_mfcc_feature[i] - debug_mfcc_feature_prev[i];
        debug_mfcc_feature_diff1[i] = debug_mfcc_feature_diff[i] - debug_mfcc_feature_diff_prev[i];
    }
    memcpy(debug_mfcc_feature_prev, debug_mfcc_feature, NUM_FEATURES * sizeof(float));
    memcpy(debug_mfcc_feature_diff_prev, debug_mfcc_feature_diff, NUM_FEATURES * sizeof(float));

    memcpy(debug_nn_features, debug_mfcc_feature, NUM_FEATURES * sizeof(float));
    memcpy(&debug_nn_features[NUM_FEATURES], debug_mfcc_feature_diff, 10 * sizeof(float));
    memcpy(&debug_nn_features[NUM_FEATURES + 10], debug_mfcc_feature_diff1, 10 * sizeof(float));

    quantize_data(debug_nn_features, debug_nn_features_q7, NUM_FEATURES + 20, 3);

    memcpy(nnom_input_data, debug_nn_features_q7, sizeof(nnom_input_data));
    model_run(debug_model);

    for (int i = 0; i < NUM_FEATURES; i++) {
        debug_band_gains[i] = (float)(nnom_output_data[i]) / 127.f;
    }

    for (int i = 0; i < NUM_FEATURES; i++) {
        debug_band_gains[i] = _MAX(debug_band_gains_prev[i] * 0.8f, debug_band_gains[i]);
    }
    memcpy(debug_band_gains_prev, debug_band_gains, NUM_FEATURES * sizeof(float));

    print_vector("debug_mfcc:", debug_mfcc_feature, NUM_FEATURES);
    print_vector("debug_nn_gains:", debug_band_gains, NUM_FEATURES);

    model_delete(debug_model);
#endif
}

static void audio_processing_task(void *arg)
{
    mfcc_t *mfcc = (mfcc_t *)arg;
    int32_t dma_audio_buffer[AUDIO_HOP_LEN] = {0};

    while (true) {
        // move buffer (50%) overlapping, move later 50% to the first 50, then fill
        memcpy(audio_buffer_16bit, &audio_buffer_16bit[AUDIO_HOP_LEN], AUDIO_HOP_LEN * sizeof(int16_t));

        if (!audio_i2s_receive(&audio_ctx, dma_audio_buffer, AUDIO_HOP_LEN, portMAX_DELAY)) {
            continue;
        }

        int64_t start_time = esp_timer_get_time();

        // convert the file to 16bit.
        for (int i = 0; i < AUDIO_HOP_LEN; i++) {
            audio_buffer_16bit[AUDIO_HOP_LEN + i] = SaturaLH((dma_audio_buffer[i] >> 8), -32768, 32767);
        }

        // get mfcc
        mfcc_compute(mfcc, audio_buffer_16bit, mfcc_feature);
        int64_t mfcc_time = esp_timer_get_time() - start_time;

        // get the first and second derivative of mfcc
        for (uint32_t i = 0; i < NUM_FEATURES; i++) {
            mfcc_feature_diff[i] = mfcc_feature[i] - mfcc_feature_prev[i];
            mfcc_feature_diff1[i] = mfcc_feature_diff[i] - mfcc_feature_diff_prev[i];
        }
        memcpy(mfcc_feature_prev, mfcc_feature, NUM_FEATURES * sizeof(float));
        memcpy(mfcc_feature_diff_prev, mfcc_feature_diff, NUM_FEATURES * sizeof(float));

        // combine MFCC with derivatives for the NN features
        memcpy(nn_features, mfcc_feature, NUM_FEATURES * sizeof(float));
        memcpy(&nn_features[NUM_FEATURES], mfcc_feature_diff, 10 * sizeof(float));
        memcpy(&nn_features[NUM_FEATURES + 10], mfcc_feature_diff1, 10 * sizeof(float));

        // quantise them using the same scale as training data (in keras), by 2^n.
        quantize_data(nn_features, nn_features_q7, NUM_FEATURES + 20, 3);

        // run the mode with the new input
        memcpy(nnom_input_data, nn_features_q7, sizeof(nnom_input_data));
        model_run(model);
        int64_t nn_time = esp_timer_get_time() - start_time - mfcc_time;

        // read the result, convert it back to float (q0.7 to float)
        for (int i = 0; i < NUM_FEATURES; i++) {
            band_gains[i] = (float)(nnom_output_data[i]) / 127.f;
        }

        // one more step, limit the change of gians, to smooth the speech, per RNNoise paper
        for (int i = 0; i < NUM_FEATURES; i++) {
            band_gains[i] = _MAX(band_gains_prev[i] * 0.8f, band_gains[i]);
        }
        memcpy(band_gains_prev, band_gains, NUM_FEATURES * sizeof(float));

        // update filter coefficient to applied dynamic gains to each frequency band
        set_gains((float *)coeff_b, (float *)b_, band_gains, NUM_FILTER, NUM_ORDER);

        // convert 16bit to float for equalizer
        for (int i = 0; i < AUDIO_HOP_LEN; i++) {
            audio_buffer[i] = audio_buffer_16bit[i + AUDIO_HOP_LEN] / 32768.f;
        }

        // finally, we apply the equalizer to this audio frame to denoise
        equalizer(audio_buffer, &audio_buffer[AUDIO_HOP_LEN], AUDIO_HOP_LEN, (float *)b_, (float *)coeff_a, NUM_FILTER,
                  NUM_ORDER);
        int64_t equalizer_time = esp_timer_get_time() - start_time - (mfcc_time + nn_time);

        // convert it back to int16
        for (int i = 0; i < AUDIO_HOP_LEN; i++) {
            audio_buffer_filtered[i] = audio_buffer[i + AUDIO_HOP_LEN] * 32768.f * 0.6f;
        }

        (void)audio_i2s_write(&audio_ctx, audio_buffer_filtered, AUDIO_HOP_LEN, portMAX_DELAY);

        int64_t total_time = esp_timer_get_time() - start_time;
        ESP_LOGI(TAG,
                 "timing us: mfcc=%" PRId64 " nn=%" PRId64 " eq=%" PRId64 " total=%" PRId64,
                 mfcc_time,
                 nn_time,
                 equalizer_time,
                 total_time);
        if (total_time > 16000) {
            ESP_LOGW(TAG, "processing overrun: %" PRId64 " us", total_time);
        }
    }
}

void app_main(void)
{
    esp_err_t err = audio_i2s_init(&audio_ctx);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "audio i2s init failed: %s", esp_err_to_name(err));
        return;
    }
    err = audio_i2s_start(&audio_ctx);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "audio i2s start failed: %s", esp_err_to_name(err));
        return;
    }

    // 26 features, 0 offset, 26 bands, 512fft, 0 preempha, attached_energy_to_band0
    mfcc_t *mfcc = mfcc_create(NUM_FEATURES, 0, NUM_FEATURES, 512, 0, true);
    if (mfcc == NULL) {
        ESP_LOGE(TAG, "failed to create mfcc");
        return;
    }

    debug_fixed_frame(mfcc);

    model = nnom_model_create();
    if (model == NULL) {
        ESP_LOGE(TAG, "failed to create model");
        return;
    }

    xTaskCreatePinnedToCore(audio_processing_task, "audio_processing", 8192, mfcc, 8, NULL, 1);
}
