#include "audio_pipeline.h"
#include <M5Unified.h>

QueueHandle_t qDetector = nullptr;
QueueHandle_t qMp3      = nullptr;
volatile uint32_t g_detectorDropCount = 0;
volatile uint32_t g_mp3DropCount      = 0;

unsigned long g_startMillis          = 0;
unsigned long g_benchmarkDurationMs  = 0;
volatile bool g_fanoutDone           = false;

static int s_sampleRate = 16000;

void audioPipelineInit(int sampleRate, int detectorQueueDepth, int mp3QueueDepth) {
  s_sampleRate = sampleRate;
  qDetector = xQueueCreate(detectorQueueDepth, sizeof(AudioChunk));
  qMp3      = xQueueCreate(mp3QueueDepth, sizeof(AudioChunk));
}

void fanoutTask(void *param) {
  AudioChunk buf;
  uint32_t seq = 0;

  for (;;) {
    if (millis() - g_startMillis > g_benchmarkDurationMs) break;

    bool ok = M5.Mic.record(buf.samples, CHUNK_SAMPLES, s_sampleRate);
    if (!ok) {
      // record() itself failed to queue (not the same as a capture
      // failure) -- back off briefly rather than busy-spin retrying.
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }
    // Tight busy-poll, no delay(): m5_mic_capture.ino's delay(1) pattern
    // was only validated at 3000ms capture granularity, where ~1ms of
    // FreeRTOS tick slop is invisible. At this 16ms chunk granularity
    // that same slop could be a meaningful fraction of a chunk period.
    // Core 1 is dedicated to this pipeline, so busy-polling here doesn't
    // steal time from anything else that matters.
    while (M5.Mic.isRecording()) { }

    buf.seq = seq++;
    buf.timestamp_us = micros();

    if (xQueueSend(qDetector, &buf, 0) != pdTRUE) g_detectorDropCount++;
    if (xQueueSend(qMp3, &buf, 0) != pdTRUE) g_mp3DropCount++;
  }

  g_fanoutDone = true;
  vTaskDelete(nullptr);
}
