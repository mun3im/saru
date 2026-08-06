#include "sd_logger.h"
#include <SD.h>
#include <SPI.h>

static constexpr gpio_num_t SD_CS_PIN = GPIO_NUM_4;

static File s_file;
static uint32_t s_bytesWritten = 0;

bool sdLoggerBegin(void) {
  return SD.begin(SD_CS_PIN, SPI, 25000000);
}

bool sdLoggerOpen(const char *filename) {
  s_file = SD.open(filename, FILE_WRITE);
  s_bytesWritten = 0;
  return (bool)s_file;
}

size_t sdLoggerWrite(const uint8_t *data, size_t len) {
  if (!s_file) return 0;
  size_t written = s_file.write(data, len);
  s_bytesWritten += written;
  return written;
}

void sdLoggerFlush(void) {
  if (s_file) s_file.flush();
}

void sdLoggerClose(void) {
  if (s_file) {
    s_file.flush();
    s_file.close();
  }
}

uint32_t sdLoggerBytesWritten(void) {
  return s_bytesWritten;
}
