#include "relume/frame_queue.hpp"

#include <utility>
#include <vector>

namespace relume {

struct FramePool::Shared {
  explicit Shared(std::size_t cap) : capacity(cap) {
    // Reserved up front so returning a frame never allocates (the deleter
    // must not throw).
    free.reserve(cap);
  }
  const std::size_t capacity;
  mutable std::mutex mutex;
  std::vector<std::unique_ptr<VideoFrame>> free;
  std::size_t outstanding = 0;
  std::uint64_t exhausted = 0;
};

FramePool::FramePool(std::size_t capacity) : shared_(std::make_shared<Shared>(capacity)) {}

FramePool::~FramePool() = default;

std::size_t FramePool::capacity() const { return shared_->capacity; }

std::size_t FramePool::outstanding() const {
  std::lock_guard lock(shared_->mutex);
  return shared_->outstanding;
}

std::uint64_t FramePool::exhausted_count() const {
  std::lock_guard lock(shared_->mutex);
  return shared_->exhausted;
}

std::shared_ptr<VideoFrame> FramePool::Acquire() {
  std::unique_ptr<VideoFrame> frame;
  {
    std::lock_guard lock(shared_->mutex);
    if (!shared_->free.empty()) {
      frame = std::move(shared_->free.back());
      shared_->free.pop_back();
    } else if (shared_->outstanding >= shared_->capacity) {
      ++shared_->exhausted;
      return nullptr;
    }
    ++shared_->outstanding;
  }
  if (!frame) {
    frame = std::make_unique<VideoFrame>();  // grows lazily up to capacity
  }
  // The deleter returns the frame to the pool, or frees it if the pool is gone.
  auto return_to_pool = [pool = std::weak_ptr<Shared>(shared_)](VideoFrame* raw) noexcept {
    std::unique_ptr<VideoFrame> owned(raw);
    const std::shared_ptr<Shared> shared = pool.lock();
    if (!shared) {
      return;  // `owned` frees the frame
    }
    std::lock_guard lock(shared->mutex);
    shared->free.push_back(std::move(owned));  // capacity reserved: no allocation
    --shared->outstanding;
  };
  return std::shared_ptr<VideoFrame>(frame.release(), std::move(return_to_pool));
}

void LatestFrameSlot::Publish(std::shared_ptr<const VideoFrame> frame) {
  {
    std::lock_guard lock(mutex_);
    if (frame_ && !taken_) {
      ++stats_.dropped_unread;
    }
    ++stats_.published;
    std::swap(frame_, frame);
    taken_ = false;
  }
  // Notifying never waits for a consumer.
  published_.notify_all();
  // `frame` now holds the replaced frame and is released here, outside the lock.
}

std::shared_ptr<const VideoFrame> LatestFrameSlot::TakeIfNew(std::uint64_t last_seen_id, std::uint64_t* generation) {
  std::lock_guard lock(mutex_);
  if (!frame_ || frame_->frame_id == last_seen_id) {
    return nullptr;
  }
  taken_ = true;
  if (generation != nullptr) {
    *generation = generation_;
  }
  return frame_;
}

std::shared_ptr<const VideoFrame> LatestFrameSlot::WaitTakeIfNew(std::uint64_t last_seen_id,
                                                                 std::uint64_t last_seen_generation,
                                                                 std::chrono::milliseconds timeout,
                                                                 std::uint64_t* generation) {
  std::unique_lock lock(mutex_);
  const std::uint64_t epoch = wake_epoch_;
  const auto is_new = [&] {
    return frame_ && (frame_->frame_id != last_seen_id || generation_ != last_seen_generation);
  };
  const bool ready = published_.wait_for(lock, timeout, [&] { return is_new() || wake_epoch_ != epoch; });
  if (!ready || !is_new()) {
    return nullptr;
  }
  taken_ = true;
  if (generation != nullptr) {
    *generation = generation_;
  }
  return frame_;
}

void LatestFrameSlot::WakeWaiters() {
  {
    std::lock_guard lock(mutex_);
    ++wake_epoch_;
  }
  published_.notify_all();
}

std::uint64_t LatestFrameSlot::generation() const {
  std::lock_guard lock(mutex_);
  return generation_;
}

std::shared_ptr<const VideoFrame> LatestFrameSlot::Latest() {
  std::lock_guard lock(mutex_);
  if (frame_) {
    taken_ = true;
  }
  return frame_;
}

void LatestFrameSlot::Clear() {
  std::shared_ptr<const VideoFrame> released;
  {
    std::lock_guard lock(mutex_);
    if (frame_ && !taken_) {
      ++stats_.dropped_unread;
    }
    std::swap(frame_, released);
    taken_ = false;
    ++generation_;
  }
}

SlotStats LatestFrameSlot::stats() const {
  std::lock_guard lock(mutex_);
  return stats_;
}

}  // namespace relume
