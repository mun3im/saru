#include "mp3_encoder.h"

extern "C" {
#include "shine_layer3.h"
}

static shine_t s_shine = nullptr;

bool mp3EncoderInit(int sampleRate, int bitrateKbps) {
  if (s_shine) {
    mp3EncoderClose();
  }

  shine_config_t config;
  shine_set_config_mpeg_defaults(&config.mpeg);
  config.wave.samplerate = sampleRate;
  config.wave.channels   = PCM_MONO;
  config.mpeg.mode       = MONO;
  config.mpeg.bitr       = bitrateKbps;

  if (shine_check_config(config.wave.samplerate, config.mpeg.bitr) < 0) {
    return false;
  }

  s_shine = shine_initialise(&config);
  return s_shine != nullptr;
}

int mp3EncoderSamplesPerPass(void) {
  if (!s_shine) return 0;
  return shine_samples_per_pass(s_shine);
}

const uint8_t *mp3EncoderEncode(const int16_t *pcm, size_t *outLen) {
  if (!s_shine) {
    *outLen = 0;
    return nullptr;
  }
  int written = 0;
  // shine_encode_buffer_interleaved takes a non-const int16_t* -- the
  // library doesn't modify its input, this is just an API const-ness
  // gap, safe to cast away.
  unsigned char *data = shine_encode_buffer_interleaved(
      s_shine, const_cast<int16_t *>(pcm), &written);
  *outLen = (size_t)written;
  return (const uint8_t *)data;
}

const uint8_t *mp3EncoderFlush(size_t *outLen) {
  if (!s_shine) {
    *outLen = 0;
    return nullptr;
  }
  int written = 0;
  unsigned char *data = shine_flush(s_shine, &written);
  *outLen = (size_t)written;
  return (const uint8_t *)data;
}

void mp3EncoderClose(void) {
  if (s_shine) {
    shine_close(s_shine);
    s_shine = nullptr;
  }
}
