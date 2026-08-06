// sd_logger.h -- SD card open/write/flush wrapper for M5Stack Core2.
//
// Pin config confirmed from source this session, not assumed: SD CS =
// GPIO4, shared SPI bus with the LCD panel (MOSI=23, MISO=38 -- note
// MISO=38, not the ESP32 VSPI default of 19 -- SCLK=18). Confirmed
// independently in M5GFX.cpp's Core2 panel-init path
// (_set_sd_spimode(bus_cfg.spi_host, GPIO_NUM_4) alongside those same
// pin_mosi/pin_miso/pin_sclk values) and M5Unified's own official
// example (examples/Advanced/Speaker_SD_wav_file/Speaker_SD_wav_file.ino:
// SDCARD_CSPIN = GPIO_NUM_4; SD.begin(SDCARD_CSPIN, SPI, 25000000)).
// M5.begin(cfg) already configures the global SPI singleton with the
// right pins via M5GFX, so sdLoggerBegin() below just needs SD.begin()
// with the CS pin -- must be called after M5.begin().

#ifndef SD_LOGGER_H
#define SD_LOGGER_H

#include <stdint.h>
#include <stddef.h>

bool sdLoggerBegin(void);
bool sdLoggerOpen(const char *filename);
size_t sdLoggerWrite(const uint8_t *data, size_t len);
void sdLoggerFlush(void);
void sdLoggerClose(void);
uint32_t sdLoggerBytesWritten(void);

#endif
