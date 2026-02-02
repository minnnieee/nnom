# ESP32-S3 rnn-denoise (ESP-IDF 5.5)

This directory contains the ESP32-S3 hardware port of the STM32 rnn-denoise reference implementation.

## Key files

- `CMakeLists.txt`: ESP-IDF project definition.
- `main/CMakeLists.txt`: Component definition that pulls in NNoM sources and MFCC.
- `main/main.c`: ESP-IDF app entry point with I2S setup and the unchanged DSP/NN pipeline.

## Build

From this directory:

```sh
idf.py set-target esp32s3
idf.py build
```
