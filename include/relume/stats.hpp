#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace relume {

// Fixed-bucket latency histogram that a capture callback or worker thread can
// record into without locks or allocation. Buckets are `BucketMicros` wide;
// values past the last bucket land in it. Percentiles report a bucket's upper
// edge, so their resolution is one bucket.
template <std::uint32_t BucketMicros, std::size_t Buckets>
class BasicLatencyHistogram {
 public:
  static constexpr double kBucketMs = BucketMicros / 1000.0;
  static constexpr std::size_t kBuckets = Buckets;

  void Record(double ms) noexcept {
    std::size_t index = 0;
    if (ms > 0.0 && std::isfinite(ms)) {
      const double bucket = std::floor(ms / kBucketMs);
      index = bucket >= static_cast<double>(kBuckets - 1) ? kBuckets - 1 : static_cast<std::size_t>(bucket);
    }
    buckets_[index].fetch_add(1, std::memory_order_relaxed);
  }

  // p in [0, 1]. Returns 0 when empty.
  double Percentile(double p) const noexcept {
    const std::uint64_t total = count();
    if (total == 0) {
      return 0.0;
    }
    const auto rank = static_cast<std::uint64_t>(std::ceil(p * static_cast<double>(total)));
    std::uint64_t seen = 0;
    for (std::size_t i = 0; i < kBuckets; ++i) {
      seen += buckets_[i].load(std::memory_order_relaxed);
      if (seen >= rank && seen > 0) {
        return static_cast<double>(i + 1) * kBucketMs;
      }
    }
    return static_cast<double>(kBuckets) * kBucketMs;
  }

  std::uint64_t count() const noexcept {
    std::uint64_t total = 0;
    for (const auto& bucket : buckets_) {
      total += bucket.load(std::memory_order_relaxed);
    }
    return total;
  }

  void Reset() noexcept {
    for (auto& bucket : buckets_) {
      bucket.store(0, std::memory_order_relaxed);
    }
  }

 private:
  std::array<std::atomic<std::uint64_t>, kBuckets> buckets_{};
};

// 0.05 ms buckets up to 50 ms: capture copies, frame intervals, render loop.
using LatencyHistogram = BasicLatencyHistogram<50, 1000>;
// 1 ms buckets up to 2 s: inference stages, depth age.
using WideLatencyHistogram = BasicLatencyHistogram<1000, 2000>;

}  // namespace relume
