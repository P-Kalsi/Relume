// Reuse policy for buffers handed to an asynchronous compositor. The
// scenarios mirror what was measured on macOS: normally the compositor holds a
// committed surface immediately; at window startup and the instant a window is
// covered, committed surfaces were never seen in use.

#include "relume/compositor_reuse.hpp"

#include <gtest/gtest.h>

namespace relume {
namespace {

TEST(CompositorReuse, NeverCommittedBufferIsReusableWhenFree) {
  CompositorReuseTracker tracker;
  EXPECT_TRUE(tracker.CanReuse(/*free=*/true, /*in_use_now=*/false, /*ever_committed=*/false, 0));
  EXPECT_FALSE(tracker.CanReuse(/*free=*/false, false, false, 0)) << "displayed, rendering or GPU-pending";
  EXPECT_FALSE(tracker.CanReuse(true, /*in_use_now=*/true, false, 0));
}

TEST(CompositorReuse, InUseIsNeverReusable) {
  CompositorReuseTracker tracker;
  tracker.ObserveInUse(10);
  EXPECT_FALSE(tracker.CanReuse(true, /*in_use_now=*/true, true, 3));
}

TEST(CompositorReuse, NotInUseIsNotEnoughForACommittedBuffer) {
  // Committed as generation 5, compositor never seen holding anything newer.
  CompositorReuseTracker tracker;
  EXPECT_FALSE(tracker.CanReuse(true, false, true, 5));
  tracker.ObserveInUse(5);  // seeing the buffer itself proves nothing about later frames
  EXPECT_FALSE(tracker.CanReuse(true, false, true, 5));
  tracker.ObserveInUse(4);  // older evidence does not count
  EXPECT_FALSE(tracker.CanReuse(true, false, true, 5));
  tracker.ObserveInUse(6);  // compositor has moved past generation 5
  EXPECT_TRUE(tracker.CanReuse(true, false, true, 5));
}

TEST(CompositorReuse, NormalDisplaySteadyState) {
  // Four buffers rotate; each commit is seen in use at once, so the previous
  // one becomes reusable as soon as the compositor releases it.
  CompositorReuseTracker tracker;
  for (std::uint64_t generation = 1; generation <= 100; ++generation) {
    tracker.ObserveInUse(generation);  // picked up immediately after commit
    if (generation > 1) {
      EXPECT_TRUE(tracker.CanReuse(true, false, true, generation - 1));
    }
    EXPECT_FALSE(tracker.CanReuse(true, false, true, generation)) << "the displayed generation itself";
  }
}

TEST(CompositorReuse, StartupOrCoverWithNothingSeenHoldsEverythingBack) {
  // Generations 1..3 committed while the compositor was not compositing the
  // window (just opened, or just covered): none may be rendered into again,
  // so the presenter runs out of buffers and skips frames instead.
  CompositorReuseTracker tracker;
  for (std::uint64_t generation = 1; generation <= 3; ++generation) {
    EXPECT_FALSE(tracker.CanReuse(true, false, true, generation));
  }
  // Compositing resumes and shows the current contents (generation 3): the
  // older ones are now safe, the current one is still not.
  tracker.ObserveInUse(3);
  EXPECT_TRUE(tracker.CanReuse(true, false, true, 1));
  EXPECT_TRUE(tracker.CanReuse(true, false, true, 2));
  EXPECT_FALSE(tracker.CanReuse(true, false, true, 3));
  EXPECT_EQ(tracker.seen_generation(), 3u);
}

}  // namespace
}  // namespace relume
