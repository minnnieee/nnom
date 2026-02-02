#include "audio_i2s.h"

#include <string.h>

#include "driver/gpio.h"
#include "esp_log.h"

#define AUDIO_I2S_RX_RING_FRAMES 8
#define AUDIO_I2S_RX_RING_BYTES (AUDIO_I2S_FRAME_SAMPLES * sizeof(int32_t) * AUDIO_I2S_RX_RING_FRAMES)
#define AUDIO_I2S_RX_TASK_STACK 4096
#define AUDIO_I2S_RX_TASK_PRIORITY 10

static const char *TAG = "audio_i2s";

static void audio_i2s_rx_task(void *arg)
{
    audio_i2s_ctx_t *ctx = (audio_i2s_ctx_t *)arg;
    int32_t rx_buffer[AUDIO_I2S_FRAME_SAMPLES] = {0};

    while (true) {
        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(ctx->rx_chan, rx_buffer, sizeof(rx_buffer), &bytes_read, portMAX_DELAY);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "i2s read failed: %s", esp_err_to_name(err));
            continue;
        }
        if (bytes_read != sizeof(rx_buffer)) {
            ESP_LOGW(TAG, "short read: %u bytes", (unsigned)bytes_read);
            continue;
        }
        if (xRingbufferSend(ctx->rx_ring, rx_buffer, bytes_read, portMAX_DELAY) != pdTRUE) {
            ESP_LOGW(TAG, "rx ringbuffer overflow");
        }
    }
}

static esp_err_t audio_i2s_init_rx(audio_i2s_ctx_t *ctx)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    esp_err_t err = i2s_new_channel(&chan_cfg, NULL, &ctx->rx_chan);
    if (err != ESP_OK) {
        return err;
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_I2S_SAMPLE_RATE),
        .slot_cfg = I2S_STD_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = GPIO_NUM_7,
            .ws = GPIO_NUM_16,
            .dout = I2S_GPIO_UNUSED,
            .din = GPIO_NUM_15,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    std_cfg.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;

    return i2s_channel_init_std(ctx->rx_chan, &std_cfg);
}

static esp_err_t audio_i2s_init_tx(audio_i2s_ctx_t *ctx)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_MASTER);
    esp_err_t err = i2s_new_channel(&chan_cfg, &ctx->tx_chan, NULL);
    if (err != ESP_OK) {
        return err;
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_I2S_SAMPLE_RATE),
        .slot_cfg = I2S_STD_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = GPIO_NUM_18,
            .ws = GPIO_NUM_17,
            .dout = GPIO_NUM_8,
            .din = I2S_GPIO_UNUSED,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    std_cfg.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;

    return i2s_channel_init_std(ctx->tx_chan, &std_cfg);
}

esp_err_t audio_i2s_init(audio_i2s_ctx_t *ctx)
{
    if (ctx == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    memset(ctx, 0, sizeof(*ctx));

    esp_err_t err = audio_i2s_init_rx(ctx);
    if (err != ESP_OK) {
        return err;
    }

    err = audio_i2s_init_tx(ctx);
    if (err != ESP_OK) {
        return err;
    }

    ctx->rx_ring = xRingbufferCreate(AUDIO_I2S_RX_RING_BYTES, RINGBUF_TYPE_BYTEBUF);
    if (ctx->rx_ring == NULL) {
        return ESP_ERR_NO_MEM;
    }

    return ESP_OK;
}

esp_err_t audio_i2s_start(audio_i2s_ctx_t *ctx)
{
    if (ctx == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    esp_err_t err = i2s_channel_enable(ctx->rx_chan);
    if (err != ESP_OK) {
        return err;
    }
    err = i2s_channel_enable(ctx->tx_chan);
    if (err != ESP_OK) {
        return err;
    }

    if (ctx->rx_task == NULL) {
        BaseType_t ok = xTaskCreatePinnedToCore(
            audio_i2s_rx_task,
            "audio_i2s_rx",
            AUDIO_I2S_RX_TASK_STACK,
            ctx,
            AUDIO_I2S_RX_TASK_PRIORITY,
            &ctx->rx_task,
            0);
        if (ok != pdPASS) {
            return ESP_ERR_NO_MEM;
        }
    }

    return ESP_OK;
}

bool audio_i2s_receive(audio_i2s_ctx_t *ctx, int32_t *dest, size_t frames, TickType_t ticks_to_wait)
{
    if (ctx == NULL || dest == NULL || frames == 0) {
        return false;
    }
    size_t bytes_needed = frames * sizeof(int32_t);
    size_t item_size = 0;
    uint8_t *item = (uint8_t *)xRingbufferReceive(ctx->rx_ring, &item_size, ticks_to_wait);
    if (item == NULL) {
        return false;
    }
    if (item_size != bytes_needed) {
        vRingbufferReturnItem(ctx->rx_ring, item);
        return false;
    }
    memcpy(dest, item, item_size);
    vRingbufferReturnItem(ctx->rx_ring, item);
    return true;
}

esp_err_t audio_i2s_write(audio_i2s_ctx_t *ctx, const int16_t *data, size_t frames, TickType_t ticks_to_wait)
{
    if (ctx == NULL || data == NULL || frames == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    size_t bytes_written = 0;
    esp_err_t err = i2s_channel_write(ctx->tx_chan, data, frames * sizeof(int16_t), &bytes_written, ticks_to_wait);
    if (err != ESP_OK) {
        return err;
    }
    return (bytes_written == frames * sizeof(int16_t)) ? ESP_OK : ESP_ERR_INVALID_SIZE;
}
