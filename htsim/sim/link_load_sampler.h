// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef LINK_LOAD_SAMPLER_H
#define LINK_LOAD_SAMPLER_H

#include <cstdint>
#include <string>

#include "config.h"

class BaseQueue;

class LinkLoadSampler {
 public:
    static bool enabled();
    static void record(BaseQueue* queue,
                       const std::string& queue_name,
                       simtime_picosec now,
                       uint64_t bytes,
                       linkspeed_bps bitrate,
                       mem_b queue_bytes);
    static void flush();
};

#endif
