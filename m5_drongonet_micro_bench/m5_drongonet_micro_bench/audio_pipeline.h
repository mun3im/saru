// audio_pipeline.h -- shared mic capture + fan-out for the dual-core
// Goertzel-vs-DrongoNet-Micro benchmark. Lives on Core 1 (see the
// .ino's task setup) so capture-side jitter never contaminates the
// Core-0 detector latency this benchmark exists to measure.
//
// CHUNK_SAMPLES=256 (16ms @ 16kHz): a clean multiple of Goertzel's
// 16-sample block (16 sub-blocks/chunk); DrongoNet's detector task
// accumulates chunks into its own 48000-sample buffer independently,
// doesn't need chunk-count alignment to that size.
//
// Two independent queues (qDetector, qMp3) so a slow consumer only
// drops its own chunks (counted, not silent) rather than blocking the
// producer or the other consumer -- this is deliberate: it's what makes
// "did detection keep up under load" a measurable signal instead of an
// assumption.

#ifndef AUDIO_PIPELINE_H
#define AUDIO_PIPELINE_H

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

#define CHUNK_SAMPLES 256

struct AudioChunk {
  int16_t  samples[CHUNK_SAMPLES];
  uint32_t seq;
  uint32_t timestamp_us;
};

extern QueueHandle_t qDetector;
extern QueueHandle_t qMp3;
extern volatile uint32_t g_detectorDropCount;
extern volatile uint32_t g_mp3DropCount;

// Shared benchmark-duration control -- set g_startMillis and
// g_benchmarkDurationMs from the .ino's setup() before spawning tasks;
// every task (fanoutTask here, plus each sketch's own detector/mp3
// tasks) checks these independently to wrap up around the same
// wall-clock time without needing tight cross-task signalling.
extern unsigned long g_startMillis;
extern unsigned long g_benchmarkDurationMs;
extern volatile bool g_fanoutDone;

// Call once from setup() before spawning any tasks.
void audioPipelineInit(int sampleRate, int detectorQueueDepth, int mp3QueueDepth);

// Task entry point -- spawn pinned to core 1. Captures continuously via
// M5.Mic (record() + tight-polled isRecording(), no delay() -- see the
// .cpp for why the delay(1) pattern proven in m5_mic_capture.ino isn't
// reused here) and pushes each completed chunk to both queues. Self-
// terminates (sets g_fanoutDone, vTaskDelete(nullptr)) once
// g_benchmarkDurationMs has elapsed since g_startMillis.
void fanoutTask(void *param);

#endif
