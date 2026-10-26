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
// Taking never removes the frame: every consumer (the renderer, the depth
// worker) keeps its own last-seen frame ID, so none can take a frame away
// from another.
//
// The slot's generation counts Clear() calls. The capture controller clears
// the slot whenever a source stops, so a frame's generation identifies the
// camera session it came from, even though every source numbers its frames
// from 1.

#include "relume/video_frame.hpp"

#include <chrono>
#include <condition_variable>
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

  // Replaces the held frame and wakes waiting consumers. Never blocks on
  // consumers.
  void Publish(std::shared_ptr<const VideoFrame> frame);

  // The newest frame if its frame_id differs from `last_seen_id`, otherwise
  // null. Sources number frames from 1, so 0 means "nothing seen yet". Marks
  // the frame as taken. If `generation` is given, it receives the slot's
  // generation for the returned frame.
  std::shared_ptr<const VideoFrame> TakeIfNew(std::uint64_t last_seen_id, std::uint64_t* generation = nullptr);

  // As TakeIfNew, but waits up to `timeout` for a frame that differs from
  // the last one seen in frame_id or in generation (a new source numbers its
  // frames from 1 again, so the ID alone can repeat). Returns null on timeout
  // or after WakeWaiters(). For a worker thread; never call it from the
  // render loop.
  std::shared_ptr<const VideoFrame> WaitTakeIfNew(std::uint64_t last_seen_id, std::uint64_t last_seen_generation,
                                                  std::chrono::milliseconds timeout,
                                                  std::uint64_t* generation = nullptr);

  // Makes every current WaitTakeIfNew return now (e.g. to stop a worker).
  void WakeWaiters();

  // Number of Clear() calls so far.
  std::uint64_t generation() const;

  // The newest frame, possibly one already taken. Marks it as taken.
  std::shared_ptr<const VideoFrame> Latest();

  // Releases the held frame, e.g. when the camera disconnects.
  void Clear();

  SlotStats stats() const;

 private:
  mutable std::mutex mutex_;
  std::condition_variable published_;
  std::shared_ptr<const VideoFrame> frame_;
  bool taken_ = false;
  std::uint64_t generation_ = 0;
  std::uint64_t wake_epoch_ = 0;  // bumped by WakeWaiters
  SlotStats stats_;
};

}  // namespace relume
