#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace relume {

// Fixed-bucket latency histogram that a capture callback can record into
// without locks or allocation. Buckets are 0.05 ms wide up to 50 ms; larger
// values land in the last bucket. Percentiles report a bucket's upper edge,
// so their resolution is 0.05 ms.
class LatencyHistogram {
 public:
  static constexpr double kBucketMs = 0.05;
  static constexpr std::size_t kBuckets = 1000;

  void Record(double ms) noexcept;
  // p in [0, 1]. Returns 0 when empty.
  double Percentile(double p) const noexcept;
  std::uint64_t count() const noexcept;
  void Reset() noexcept;

 private:
  std::array<std::atomic<std::uint64_t>, kBuckets> buckets_{};
};

}  // namespace relume
