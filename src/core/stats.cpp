#include "relume/stats.hpp"

#include <cmath>

namespace relume {

void LatencyHistogram::Record(double ms) noexcept {
  std::size_t index = 0;
  if (ms > 0.0 && std::isfinite(ms)) {
    const double bucket = std::floor(ms / kBucketMs);
    index = bucket >= static_cast<double>(kBuckets - 1) ? kBuckets - 1 : static_cast<std::size_t>(bucket);
  }
  buckets_[index].fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t LatencyHistogram::count() const noexcept {
  std::uint64_t total = 0;
  for (const auto& bucket : buckets_) {
    total += bucket.load(std::memory_order_relaxed);
  }
  return total;
}

double LatencyHistogram::Percentile(double p) const noexcept {
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

void LatencyHistogram::Reset() noexcept {
  for (auto& bucket : buckets_) {
    bucket.store(0, std::memory_order_relaxed);
  }
}

}  // namespace relume
