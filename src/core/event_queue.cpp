#include "relume/event_queue.hpp"

#include <utility>

namespace relume {

void CameraEventQueue::Post(CameraEvent event) {
  std::lock_guard lock(mutex_);
  if (capacity_ == 0) {
    ++discarded_;
    return;
  }
  if (events_.size() >= capacity_) {
    events_.pop_front();
    ++discarded_;
  }
  events_.push_back(std::move(event));
}

std::optional<CameraEvent> CameraEventQueue::Poll() {
  std::lock_guard lock(mutex_);
  if (events_.empty()) {
    return std::nullopt;
  }
  CameraEvent event = std::move(events_.front());
  events_.pop_front();
  return event;
}

std::uint64_t CameraEventQueue::discarded() const {
  std::lock_guard lock(mutex_);
  return discarded_;
}

}  // namespace relume
