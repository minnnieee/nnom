/*
 * ESP32-S3 (ESP-IDF 5.5) port of the STM32 rnn-denoise reference implementation.
 *
 * DSP + NN logic is kept byte-equivalent where possible. Only HAL/BSP/peripheral
 * layers are replaced with ESP-IDF drivers.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_timer.h"

#include "nnom.h"
#include "denoise_weights.h"
#include "mfcc.h"

#include "equalizer_coeff.h"

#define NUM_FEATURES NUM_FILTER

#define _MAX(x, y) (((x) > (y)) ? (x) : (y))
#define _MIN(x, y) (((x) < (y)) ? (x) : (y))

#define NUM_CHANNELS    1
#define SAMPLE_RATE     16000
#define AUDIO_FRAME_LEN 512

#define I2S_MIC_BCLK GPIO_NUM_7
#define I2S_MIC_LRCK GPIO_NUM_16
#define I2S_MIC_DOUT GPIO_NUM_15

#define I2S_SPK_BCLK GPIO_NUM_18
#define I2S_SPK_LRCK GPIO_NUM_17
#define I2S_SPK_DIN  GPIO_NUM_8

#define LED_GPIO GPIO_NUM_2

// audio buffer for input
static float audio_buffer[AUDIO_FRAME_LEN] = {0};
static int16_t audio_buffer_16bit[AUDIO_FRAME_LEN] = {0};

// buffer for output
static int16_t audio_buffer_filtered[AUDIO_FRAME_LEN / 2] = {0};

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

// nnom model
static nnom_model_t *model;

// audio DMA-style buffer (mirrors STM32 half/full behavior)
static int32_t dma_audio_buffer[AUDIO_FRAME_LEN];

static i2s_chan_handle_t i2s_rx_handle;
static i2s_chan_handle_t i2s_tx_handle;

static volatile bool is_half_updated = false;
static volatile bool is_full_updated = false;

static void y_h_update(float *y_h, uint32_t len)
{
    for (uint32_t i = len - 1; i > 0; i--)
        y_h[i] = y_h[i - 1];
}

// equalizer by multiple n order iir band pass filter.
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
    for (uint32_t i = 0; i < num_coeff; i++)
    {
        y[i] = 0;
        for (uint32_t n = 0; n < num_band; n++)
        {
            y_h_update(y_h[n], num_coeff);
            y_h[n][0] = b[n * num_coeff] * x_h[i + num_coeff];
            for (uint32_t c = 1; c < num_coeff; c++)
                y_h[n][0] += b[n * num_coeff + c] * x_h[num_coeff + i - c] - a[n * num_coeff + c] * y_h[n][c];
            y[i] += y_h[n][0];
        }
    }
    // store the x for the state of next round
    memcpy(x_h, &x[signal_len - num_coeff], num_coeff * sizeof(float));

    // i > num_coeff; the rest data not involed the x history
    for (uint32_t i = num_coeff; i < signal_len; i++)
    {
        y[i] = 0;
        for (uint32_t n = 0; n < num_band; n++)
        {
            y_h_update(y_h[n], num_coeff);
            y_h[n][0] = b[n * num_coeff] * x[i];
            for (uint32_t c = 1; c < num_coeff; c++)
                y_h[n][0] += b[n * num_coeff + c] * x[i - c] - a[n * num_coeff + c] * y_h[n][c];
            y[i] += y_h[n][0];
        }
    }
}

// set dynamic gains. Multiple gains x b_coeff
static void set_gains(float *b_in, float *b_out, float *gains, uint32_t num_band, uint32_t num_order)
{
    uint32_t num_coeff = num_order * 2 + 1;
    for (uint32_t i = 0; i < num_band; i++)
        for (uint32_t c = 0; c < num_coeff; c++)
            b_out[num_coeff * i + c] = b_in[num_coeff * i + c] * gains[i];
}

static void quantize_data(float *din, int8_t *dout, uint32_t size, uint32_t int_bit)
{
    float limit = (1 << int_bit);
    for (uint32_t i = 0; i < size; i++)
        dout[i] = (int8_t)(_MAX(_MIN(din[i], limit), -limit) / limit * 127);
}

static void led_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = 1ULL << LED_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
}

static void led_set(bool on)
{
    gpio_set_level(LED_GPIO, on ? 1 : 0);
}

static uint32_t us_timer_get(void)
{
    return (uint32_t)esp_timer_get_time();
}

static void i2s_init(void)
{
    i2s_chan_config_t rx_chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    rx_chan_cfg.auto_clear = true;
    ESP_ERROR_CHECK(i2s_new_channel(&rx_chan_cfg, NULL, &i2s_rx_handle));

    i2s_chan_config_t tx_chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_MASTER);
    tx_chan_cfg.auto_clear = true;
    ESP_ERROR_CHECK(i2s_new_channel(&tx_chan_cfg, &i2s_tx_handle, NULL));

    i2s_std_config_t rx_std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_MIC_BCLK,
            .ws = I2S_MIC_LRCK,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_MIC_DOUT,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    rx_std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    i2s_std_config_t tx_std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_SPK_BCLK,
            .ws = I2S_SPK_LRCK,
            .dout = I2S_SPK_DIN,
            .din = I2S_GPIO_UNUSED,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(i2s_rx_handle, &rx_std_cfg));
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(i2s_tx_handle, &tx_std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(i2s_rx_handle));
    ESP_ERROR_CHECK(i2s_channel_enable(i2s_tx_handle));
}

static void i2s_rx_task(void *arg)
{
    size_t bytes_read = 0;

    while (true)
    {
        ESP_ERROR_CHECK(i2s_channel_read(
            i2s_rx_handle,
            &dma_audio_buffer[0],
            sizeof(int32_t) * (AUDIO_FRAME_LEN / 2),
            &bytes_read,
            portMAX_DELAY));
        if (bytes_read == sizeof(int32_t) * (AUDIO_FRAME_LEN / 2))
        {
            is_half_updated = true;
        }

        ESP_ERROR_CHECK(i2s_channel_read(
            i2s_rx_handle,
            &dma_audio_buffer[AUDIO_FRAME_LEN / 2],
            sizeof(int32_t) * (AUDIO_FRAME_LEN / 2),
            &bytes_read,
            portMAX_DELAY));
        if (bytes_read == sizeof(int32_t) * (AUDIO_FRAME_LEN / 2))
        {
            is_full_updated = true;
        }
    }
}

static void i2s_tx_write_mono(const int16_t *mono_samples, size_t sample_count)
{
    int16_t tx_buffer[AUDIO_FRAME_LEN] = {0};
    size_t bytes_written = 0;

    for (size_t i = 0; i < sample_count; i++)
    {
        tx_buffer[i * 2] = mono_samples[i];
        tx_buffer[i * 2 + 1] = mono_samples[i];
    }

    ESP_ERROR_CHECK(i2s_channel_write(i2s_tx_handle, tx_buffer, sample_count * sizeof(int16_t) * 2, &bytes_written, portMAX_DELAY));
}

void app_main(void)
{
    #define SaturaLH(N, L, H) (((N) < (L)) ? (L) : (((N) > (H)) ? (H) : (N)))

    uint32_t start_time = 0;
    uint32_t mfcc_time = 0;
    uint32_t nn_time = 0;
    uint32_t equalizer_time = 0;

    int32_t *p_new_data;

    model = nnom_model_create();

    // mfcc features, 0 offset, 26 bands, 512fft, 0 preempha, attached_energy_to_band0
    mfcc_t *mfcc = mfcc_create(NUM_FEATURES, 0, NUM_FEATURES, 512, 0, true);

    led_init();
    i2s_init();

    xTaskCreatePinnedToCore(i2s_rx_task, "i2s_rx_task", 4096, NULL, configMAX_PRIORITIES - 1, NULL, 0);

    while (true)
    {
        // move buffer (50%) overlapping, move later 50% to the first 50, then fill
        memcpy(audio_buffer_16bit, &audio_buffer_16bit[AUDIO_FRAME_LEN / 2], AUDIO_FRAME_LEN / 2 * sizeof(int16_t));

        // wait for new data
        while (!is_half_updated && !is_full_updated)
            ;

        start_time = us_timer_get();

        if (is_half_updated)
        {
            is_half_updated = false;
            p_new_data = dma_audio_buffer;
        }
        else
        {
            is_full_updated = false;
            p_new_data = &dma_audio_buffer[AUDIO_FRAME_LEN / 2];
        }

        // convert the file to 16bit.
        for (int i = 0; i < AUDIO_FRAME_LEN / 2; i++)
            audio_buffer_16bit[AUDIO_FRAME_LEN / 2 + i] = SaturaLH((p_new_data[i] >> 8), -32768, 32767);

        // get mfcc
        mfcc_compute(mfcc, audio_buffer_16bit, mfcc_feature);
        mfcc_time = us_timer_get() - start_time;

        // get the first and second derivative of mfcc
        for (uint32_t i = 0; i < NUM_FEATURES; i++)
        {
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
        nn_time = us_timer_get() - start_time - mfcc_time;

        // read the result, convert it back to float (q0.7 to float)
        for (int i = 0; i < NUM_FEATURES; i++)
            band_gains[i] = (float)(nnom_output_data[i]) / 127.f;

        // one more step, limit the change of gians, to smooth the speech, per RNNoise paper
        for (int i = 0; i < NUM_FEATURES; i++)
            band_gains[i] = _MAX(band_gains_prev[i] * 0.8f, band_gains[i]);
        memcpy(band_gains_prev, band_gains, NUM_FEATURES * sizeof(float));

        // update filter coefficient to applied dynamic gains to each frequency band
        set_gains((float *)coeff_b, (float *)b_, band_gains, NUM_FILTER, NUM_ORDER);

        // convert 16bit to float for equalizer
        for (int i = 0; i < AUDIO_FRAME_LEN / 2; i++)
            audio_buffer[i] = audio_buffer_16bit[i + AUDIO_FRAME_LEN / 2] / 32768.f;

        // finally, we apply the equalizer to this audio frame to denoise
        equalizer(audio_buffer, &audio_buffer[AUDIO_FRAME_LEN / 2], AUDIO_FRAME_LEN / 2, (float *)b_, (float *)coeff_a, NUM_FILTER, NUM_ORDER);
        equalizer_time = us_timer_get() - start_time - (mfcc_time + nn_time);

        // convert it back to int16
        for (int i = 0; i < AUDIO_FRAME_LEN / 2; i++)
            audio_buffer_filtered[i] = audio_buffer[i + AUDIO_FRAME_LEN / 2] * 32768.f * 0.6f; // 0.7 is the filter band overlapping factor

        // voice detection to show an LED
        if (nnom_output_data1[0] >= 64)
            led_set(true);
        else
            led_set(false);

        i2s_tx_write_mono(audio_buffer_filtered, AUDIO_FRAME_LEN / 2);

        (void)equalizer_time;
        (void)nn_time;
    }
}
