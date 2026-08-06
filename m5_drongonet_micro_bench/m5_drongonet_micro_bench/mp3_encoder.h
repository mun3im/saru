// mp3_encoder.h -- thin Arduino-facing wrapper around vendored Shine
// (libshine, LGPLv2, github.com/toots/shine), the standard lightweight
// fixed-point MP3 encoder used in embedded/ESP32 hobbyist projects. No
// MP3 encoder exists in this repo or as an Arduino Library Manager
// package for ESP32, hence vendoring rather than a library dependency.
//
// 16kHz forces MPEG-2 LSF framing (not MPEG-1): confirmed via the
// vendored source (shine_types.h: GRANULE_SIZE=576,
// granules_per_frame[MPEG_II]=1 -> shine_samples_per_pass() returns 576
// for a 16kHz config, not the 1152 MPEG-1 would use). Callers must feed
// exactly mp3EncoderSamplesPerPass() samples per encode() call -- the
// Core-1 encoder task owns a small PCM staging ring buffer to re-chunk
// this benchmark's 256-sample audio chunks into that exact size.

#ifndef MP3_ENCODER_H
#define MP3_ENCODER_H

#include <stdint.h>
#include <stddef.h>

// sampleRate must be one of Shine's supported rates (16000 here).
// bitrateKbps must be a supported MPEG-II bitrate (8-160, see
// shine_layer3.h's bitrates table comment). Returns false if the
// sample rate / bitrate combination isn't supported, or allocation
// failed.
bool mp3EncoderInit(int sampleRate, int bitrateKbps);

// Samples required per encode() call. Expect 576 at 16kHz. Returns 0
// if not initialised.
int mp3EncoderSamplesPerPass(void);

// Encodes exactly mp3EncoderSamplesPerPass() mono PCM samples. Returned
// pointer is owned by the encoder and only valid until the next
// encode()/flush()/close() call -- caller must copy out before that.
// Returns nullptr / *outLen=0 if nothing was written this call (Shine
// buffers internally across calls, doesn't emit a frame every call).
const uint8_t *mp3EncoderEncode(const int16_t *pcm, size_t *outLen);

// Flush any buffered-but-unencoded data. Call once before close().
const uint8_t *mp3EncoderFlush(size_t *outLen);

void mp3EncoderClose(void);

#endif
