#pragma once

// When may a window buffer that was handed to an asynchronous compositor
// (Core Animation on macOS) be rendered into again?
//
// The compositor reads committed buffers some time after the commit and
// reports that only indirectly (on macOS: IOSurfaceIsInUse). "Not in use
// right now" is not enough on its own: a buffer the compositor has not picked
// up yet is also "not in use". So a previously committed buffer is reusable
// only once the compositor has been seen holding a NEWER committed buffer,
// i.e. it has moved past this one. Buffers never committed are reusable as
// soon as they are free (the GPU has finished with them).
//
// Assumption: the compositor applies commits in order and never returns to
// superseded contents. Generations must increase with every commit.

#include <cstdint>

namespace relume {

class CompositorReuseTracker {
 public:
  // A committed buffer of this generation was observed in use by the compositor.
  void ObserveInUse(std::uint64_t generation) {
    if (generation > seen_generation_) {
      seen_generation_ = generation;
    }
  }

  // `free`: not displayed, not being rendered, GPU finished with it.
  // `in_use_now`: the compositor reports it is using the buffer.
  // `ever_committed` / `generation`: whether, and as which generation, it was
  // last given to the compositor.
  bool CanReuse(bool free, bool in_use_now, bool ever_committed, std::uint64_t generation) const {
    if (!free || in_use_now) {
      return false;
    }
    return !ever_committed || seen_generation_ > generation;
  }

  std::uint64_t seen_generation() const { return seen_generation_; }

 private:
  std::uint64_t seen_generation_ = 0;
};

}  // namespace relume
