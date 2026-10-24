#pragma once

// Bounded frame handoff between a camera source (producer) and its consumers.
//
// FramePool owns at most `capacity` VideoFrame buffers and recycles them: a
// frame returns to the pool when its last shared_ptr is released. A frame may
// outlive the pool; it is then simply freed. The producer never waits: when
// every buffer is in use, Acquire() returns null and the frame is dropped.
//
// LatestFrameSlot keeps only the newest published frame. Consumers take
// references to it; a frame replaced before anyone took it counts as dropped.
// Locks are held only for pointer swaps, and frames are released outside them.

#include "relume/video_frame.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace relume {

class FramePool {
 public:
  explicit FramePool(std::size_t capacity);
  ~FramePool();
  FramePool(const FramePool&) = delete;
  FramePool& operator=(const FramePool&) = delete;

  // A frame for the producer to fill, or null if all buffers are in use. The
  // frame keeps the size and data capacity it had when last used.
  std::shared_ptr<VideoFrame> Acquire();

  std::size_t capacity() const;
  std::size_t outstanding() const;       // frames currently handed out
  std::uint64_t exhausted_count() const;  // Acquire() calls that returned null

 private:
  struct Shared;
  std::shared_ptr<Shared> shared_;
};

struct SlotStats {
  std::uint64_t published = 0;
  std::uint64_t dropped_unread = 0;  // replaced or cleared before any consumer took it
};

class LatestFrameSlot {
 public:
  LatestFrameSlot() = default;
  LatestFrameSlot(const LatestFrameSlot&) = delete;
  LatestFrameSlot& operator=(const LatestFrameSlot&) = delete;

  // Replaces the held frame. Never blocks on consumers.
  void Publish(std::shared_ptr<const VideoFrame> frame);

  // The newest frame if its frame_id differs from `last_seen_id`, otherwise
  // null. Sources number frames from 1, so 0 means "nothing seen yet". Marks
  // the frame as taken.
  std::shared_ptr<const VideoFrame> TakeIfNew(std::uint64_t last_seen_id);

  // The newest frame, possibly one already taken. Marks it as taken.
  std::shared_ptr<const VideoFrame> Latest();

  // Releases the held frame, e.g. when the camera disconnects.
  void Clear();

  SlotStats stats() const;

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<const VideoFrame> frame_;
  bool taken_ = false;
  SlotStats stats_;
};

}  // namespace relume
