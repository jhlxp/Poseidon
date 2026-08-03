// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "link_load_sampler.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

struct Endpoint {
    std::string type = "unknown";
    int id = -1;
};

struct LinkMeta {
    std::string name;
    std::string layer = "unknown";
    std::string direction = "unknown";
    Endpoint src;
    Endpoint dst;
    int bundle = -1;
    double rate_gbps = 0.0;
};

struct LinkState {
    int id = -1;
    LinkMeta meta;
    uint64_t bucket = 0;
    uint64_t bytes = 0;
    uint64_t max_queue_bytes = 0;
    bool initialized = false;
};

struct SamplerState {
    bool configured = false;
    bool is_enabled = false;
    bool registered_atexit = false;
    bool flushed = false;
    simtime_picosec interval_ps = timeFromUs(1000.0);
    std::ofstream info;
    std::ofstream samples;
    std::unordered_map<BaseQueue*, LinkState> links;
    int next_link_id = 0;
};

SamplerState& state() {
    static SamplerState sampler;
    return sampler;
}

bool truthy(const char* value) {
    if (!value) {
        return false;
    }
    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

int parse_int_after_prefix(const std::string& text, const std::string& prefix) {
    if (text.rfind(prefix, 0) != 0) {
        return -1;
    }
    size_t pos = prefix.size();
    while (pos < text.size() && text[pos] == '_') {
        ++pos;
    }
    size_t begin = pos;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    if (begin == pos) {
        return -1;
    }
    return std::atoi(text.substr(begin, pos - begin).c_str());
}

Endpoint parse_endpoint(const std::string& endpoint) {
    Endpoint out;
    int id = parse_int_after_prefix(endpoint, "SRC");
    if (id >= 0) {
        out.type = "host";
        out.id = id;
        return out;
    }
    id = parse_int_after_prefix(endpoint, "LOCAL_SRC");
    if (id >= 0) {
        out.type = "host";
        out.id = id;
        return out;
    }
    id = parse_int_after_prefix(endpoint, "DST");
    if (id >= 0) {
        out.type = "host";
        out.id = id;
        return out;
    }
    id = parse_int_after_prefix(endpoint, "LS");
    if (id >= 0) {
        out.type = "tor";
        out.id = id;
        return out;
    }
    id = parse_int_after_prefix(endpoint, "US");
    if (id >= 0) {
        out.type = "agg";
        out.id = id;
        return out;
    }
    id = parse_int_after_prefix(endpoint, "CS");
    if (id >= 0) {
        out.type = "core";
        out.id = id;
        return out;
    }
    id = parse_int_after_prefix(endpoint, "L0");
    if (id >= 0) {
        out.type = "oxc_l0";
        out.id = id;
        return out;
    }
    id = parse_int_after_prefix(endpoint, "L1");
    if (id >= 0) {
        out.type = "oxc_l1";
        out.id = id;
        return out;
    }
    id = parse_int_after_prefix(endpoint, "OCS");
    if (id >= 0) {
        out.type = "oxc_ocs";
        out.id = id;
        return out;
    }
    return out;
}

std::string strip_bundle(const std::string& name, int* bundle) {
    *bundle = -1;
    size_t open = name.find('(');
    if (open == std::string::npos) {
        return name;
    }
    size_t close = name.find(')', open + 1);
    if (close != std::string::npos) {
        *bundle = std::atoi(name.substr(open + 1, close - open - 1).c_str());
    }
    return name.substr(0, open);
}

void classify(LinkMeta* meta) {
    int bundle = -1;
    std::string clean = strip_bundle(meta->name, &bundle);
    meta->bundle = bundle;

    size_t arrow = clean.find("->");
    if (arrow == std::string::npos) {
        return;
    }

    meta->src = parse_endpoint(clean.substr(0, arrow));
    meta->dst = parse_endpoint(clean.substr(arrow + 2));

    const std::string& s = meta->src.type;
    const std::string& d = meta->dst.type;
    if ((s == "host" && d == "tor") || (s == "tor" && d == "host")) {
        meta->layer = "host_tor";
    } else if (s == "host" && d == "host") {
        meta->layer = "oxc_local_direct";
    } else if ((s == "tor" && d == "agg") || (s == "agg" && d == "tor")) {
        meta->layer = "tor_agg";
    } else if ((s == "agg" && d == "core") || (s == "core" && d == "agg")) {
        meta->layer = "agg_core";
    } else if ((s == "host" && d == "oxc_l0") || (s == "oxc_l0" && d == "host")) {
        meta->layer = "oxc_host_l0";
    } else if ((s == "oxc_l0" && d == "oxc_l1") || (s == "oxc_l1" && d == "oxc_l0")) {
        meta->layer = "oxc_l0_l1";
    } else if ((s == "oxc_l1" && d == "oxc_l1")
               || (s == "oxc_l1" && d == "oxc_ocs")
               || (s == "oxc_ocs" && d == "oxc_l1")) {
        meta->layer = "oxc_l1_ocs";
    }

    if ((s == "host" && d == "tor") || (s == "tor" && d == "agg") || (s == "agg" && d == "core")) {
        meta->direction = "up";
    } else if (s == "host" && d == "host") {
        meta->direction = "local";
    } else if ((s == "core" && d == "agg") || (s == "agg" && d == "tor") || (s == "tor" && d == "host")) {
        meta->direction = "down";
    } else if ((s == "host" && d == "oxc_l0")
               || (s == "oxc_l0" && d == "oxc_l1")) {
        meta->direction = "up";
    } else if ((s == "oxc_l1" && d == "oxc_l0")
               || (s == "oxc_l0" && d == "host")) {
        meta->direction = "down";
    } else if ((s == "oxc_l1" && d == "oxc_l1")
               || (s == "oxc_l1" && d == "oxc_ocs")
               || (s == "oxc_ocs" && d == "oxc_l1")) {
        meta->direction = "cross";
    }
}

void ensure_configured() {
    SamplerState& s = state();
    if (s.configured) {
        return;
    }
    s.configured = true;
    s.is_enabled = truthy(std::getenv("HTSIM_LINK_LOAD_SAMPLE"));
    if (!s.is_enabled) {
        return;
    }

    if (const char* interval_us = std::getenv("HTSIM_LINK_LOAD_SAMPLE_US")) {
        double us = std::atof(interval_us);
        if (us > 0.0) {
            s.interval_ps = timeFromUs(us);
        }
    }

    std::filesystem::create_directories("output_metrics");
    s.info.open("output_metrics/link_info.csv", std::ios::out | std::ios::trunc);
    s.samples.open("output_metrics/link_load_1ms.csv", std::ios::out | std::ios::trunc);
    s.info << "link_id,link_name,layer,direction,src_type,src_id,dst_type,dst_id,bundle,rate_gbps\n";
    s.samples << "time_ms,bucket,link_id,bytes,throughput_gbps,max_queue_bytes\n";

    if (!s.registered_atexit) {
        std::atexit(LinkLoadSampler::flush);
        s.registered_atexit = true;
    }
}

void write_bucket(LinkState& link, uint64_t bucket) {
    SamplerState& s = state();
    if (!s.samples.is_open() || (link.bytes == 0 && link.max_queue_bytes == 0)) {
        return;
    }
    const double time_ms = static_cast<double>(bucket * s.interval_ps) / 1e9;
    const double throughput_gbps = static_cast<double>(link.bytes) * 8000.0 /
                                   static_cast<double>(s.interval_ps);
    s.samples << time_ms << "," << bucket << "," << link.id << "," << link.bytes
              << "," << throughput_gbps << "," << link.max_queue_bytes << "\n";
}

LinkState& get_or_create_link(BaseQueue* queue,
                              const std::string& queue_name,
                              linkspeed_bps bitrate) {
    SamplerState& s = state();
    auto it = s.links.find(queue);
    if (it != s.links.end()) {
        return it->second;
    }

    LinkState link;
    link.id = s.next_link_id++;
    link.meta.name = queue_name.empty() ? ("queue_" + std::to_string(link.id)) : queue_name;
    link.meta.rate_gbps = speedAsGbps(bitrate);
    classify(&link.meta);

    if (s.info.is_open()) {
        const LinkMeta& m = link.meta;
        s.info << link.id << "," << m.name << "," << m.layer << "," << m.direction
               << "," << m.src.type << "," << m.src.id
               << "," << m.dst.type << "," << m.dst.id
               << "," << m.bundle << "," << m.rate_gbps << "\n";
    }

    auto inserted = s.links.emplace(queue, std::move(link));
    return inserted.first->second;
}

}  // namespace

bool LinkLoadSampler::enabled() {
    ensure_configured();
    return state().is_enabled;
}

void LinkLoadSampler::record(BaseQueue* queue,
                             const std::string& queue_name,
                             simtime_picosec now,
                             uint64_t bytes,
                             linkspeed_bps bitrate,
                             mem_b queue_bytes) {
    ensure_configured();
    SamplerState& s = state();
    if (!s.is_enabled || bytes == 0 || s.interval_ps == 0) {
        return;
    }

    LinkState& link = get_or_create_link(queue, queue_name, bitrate);
    const uint64_t bucket = now / s.interval_ps;
    if (!link.initialized) {
        link.bucket = bucket;
        link.initialized = true;
    } else if (bucket != link.bucket) {
        write_bucket(link, link.bucket);
        link.bucket = bucket;
        link.bytes = 0;
        link.max_queue_bytes = 0;
    }

    link.bytes += bytes;
    if (queue_bytes > 0) {
        link.max_queue_bytes = std::max<uint64_t>(link.max_queue_bytes,
                                                  static_cast<uint64_t>(queue_bytes));
    }
}

void LinkLoadSampler::flush() {
    SamplerState& s = state();
    if (!s.configured || !s.is_enabled || s.flushed) {
        return;
    }
    for (auto& entry : s.links) {
        LinkState& link = entry.second;
        if (link.initialized) {
            write_bucket(link, link.bucket);
            link.bytes = 0;
            link.max_queue_bytes = 0;
        }
    }
    if (s.info.is_open()) {
        s.info.flush();
    }
    if (s.samples.is_open()) {
        s.samples.flush();
    }
    s.flushed = true;
}
