// FramePool / LatestFrameSlot behaviour: bounded memory, newest-wins, drop
// accounting, recycling, and safe ownership when frames outlive the pool.
// The threaded tests are meant to be run under the tsan and asan presets too.

#include "relume/frame_queue.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <set>
#include <thread>
#include <vector>

namespace relume {
namespace {

std::shared_ptr<const VideoFrame> MakeFrame(FramePool& pool, std::uint64_t id) {
  std::shared_ptr<VideoFrame> frame = pool.Acquire();
  if (frame) {
    frame->frame_id = id;
    AllocateFrame(*frame, PixelFormat::kBgra8, 8, 8);
  }
  return frame;
}

TEST(FrameQueue, NewestFrameWins) {
  FramePool pool(4);
  LatestFrameSlot slot;
  slot.Publish(MakeFrame(pool, 1));
  slot.Publish(MakeFrame(pool, 2));
  slot.Publish(MakeFrame(pool, 3));
  const auto latest = slot.Latest();
  ASSERT_TRUE(latest);
  EXPECT_EQ(latest->frame_id, 3u);
  EXPECT_EQ(slot.stats().published, 3u);
  EXPECT_EQ(slot.stats().dropped_unread, 2u);
}

TEST(FrameQueue, TakenFramesAreNotCountedAsDropped) {
  FramePool pool(4);
  LatestFrameSlot slot;
  slot.Publish(MakeFrame(pool, 1));
  EXPECT_TRUE(slot.TakeIfNew(0));
  slot.Publish(MakeFrame(pool, 2));
  EXPECT_EQ(slot.stats().dropped_unread, 0u);
}

TEST(FrameQueue, TakeIfNewSkipsAlreadySeenFrame) {
  FramePool pool(2);
  LatestFrameSlot slot;
  EXPECT_FALSE(slot.TakeIfNew(0)) << "empty slot";
  slot.Publish(MakeFrame(pool, 1));
  const auto first = slot.TakeIfNew(0);
  ASSERT_TRUE(first);
  EXPECT_FALSE(slot.TakeIfNew(first->frame_id));
  slot.Publish(MakeFrame(pool, 2));
  const auto second = slot.TakeIfNew(first->frame_id);
  ASSERT_TRUE(second);
  EXPECT_EQ(second->frame_id, 2u);
}

TEST(FrameQueue, ExhaustedPoolDropsWithoutAllocating) {
  FramePool pool(3);
  std::vector<std::shared_ptr<VideoFrame>> held;
  std::set<const VideoFrame*> addresses;
  for (int i = 0; i < 3; ++i) {
    held.push_back(pool.Acquire());
    ASSERT_TRUE(held.back());
    addresses.insert(held.back().get());
  }
  EXPECT_FALSE(pool.Acquire());
  EXPECT_FALSE(pool.Acquire());
  EXPECT_EQ(pool.exhausted_count(), 2u);
  EXPECT_EQ(pool.outstanding(), 3u);

  // Releasing one makes exactly that buffer available again.
  const VideoFrame* released = held.back().get();
  held.pop_back();
  const auto again = pool.Acquire();
  ASSERT_TRUE(again);
  EXPECT_EQ(again.get(), released);
  EXPECT_EQ(addresses.count(again.get()), 1u);
}

TEST(FrameQueue, RecycledFramesKeepTheirStorage) {
  FramePool pool(1);
  const std::byte* storage = nullptr;
  {
    auto frame = pool.Acquire();
    ASSERT_TRUE(frame);
    AllocateFrame(*frame, PixelFormat::kBgra8, 64, 64);
    storage = frame->data.data();
  }
  EXPECT_EQ(pool.outstanding(), 0u);
  auto frame = pool.Acquire();
  ASSERT_TRUE(frame);
  AllocateFrame(*frame, PixelFormat::kBgra8, 64, 64);
  EXPECT_EQ(frame->data.data(), storage) << "same size must not reallocate";
}

TEST(FrameQueue, MultipleConsumersKeepFrameAlive) {
  FramePool pool(2);
  LatestFrameSlot slot;
  slot.Publish(MakeFrame(pool, 1));
  auto render = slot.Latest();
  auto other = slot.Latest();
  slot.Publish(MakeFrame(pool, 2));
  EXPECT_EQ(pool.outstanding(), 2u);
  EXPECT_EQ(render->frame_id, 1u);
  render.reset();
  EXPECT_EQ(pool.outstanding(), 2u) << "second consumer still holds frame 1";
  other.reset();
  EXPECT_EQ(pool.outstanding(), 1u);
}

TEST(FrameQueue, ClearReleasesHeldFrame) {
  FramePool pool(1);
  LatestFrameSlot slot;
  slot.Publish(MakeFrame(pool, 1));
  EXPECT_EQ(pool.outstanding(), 1u);
  slot.Clear();
  EXPECT_EQ(pool.outstanding(), 0u);
  EXPECT_FALSE(slot.Latest());
  EXPECT_EQ(slot.stats().dropped_unread, 1u);
}

TEST(FrameQueue, FramesMayOutliveThePool) {
  std::shared_ptr<const VideoFrame> survivor;
  auto slot = std::make_unique<LatestFrameSlot>();
  {
    FramePool pool(2);
    slot->Publish(MakeFrame(pool, 7));
    survivor = MakeFrame(pool, 8);
  }
  // The pool is gone; releasing these must free them (checked by ASan).
  EXPECT_EQ(survivor->frame_id, 8u);
  EXPECT_EQ(slot->Latest()->frame_id, 7u);
  survivor.reset();
  slot.reset();
}

TEST(FrameQueue, SlotMayBeDestroyedBeforeThePool) {
  FramePool pool(2);
  {
    LatestFrameSlot slot;
    slot.Publish(MakeFrame(pool, 1));
  }
  EXPECT_EQ(pool.outstanding(), 0u);
}

// One producer publishing as fast as possible, two consumers taking frames.
// Checks that each consumer sees strictly increasing IDs, that the pool never
// hands out more than its capacity, and that everything is returned at the end.
TEST(FrameQueue, ConcurrentProducerAndConsumers) {
  constexpr std::size_t kCapacity = 4;
  constexpr std::uint64_t kFrames = 20000;
  FramePool pool(kCapacity);
  LatestFrameSlot slot;
  std::atomic<bool> producer_done{false};
  std::atomic<bool> order_ok{true};
  std::atomic<std::uint64_t> taken{0};

  auto consume = [&] {
    std::uint64_t last = 0;
    std::shared_ptr<const VideoFrame> holding;
    while (!producer_done.load()) {
      auto frame = slot.TakeIfNew(last);
      if (!frame) {
        std::this_thread::yield();
        continue;
      }
      if (frame->frame_id <= last) {
        order_ok = false;
      }
      last = frame->frame_id;
      taken.fetch_add(1);
      holding = std::move(frame);  // hold one frame across iterations, like a renderer
    }
  };
  std::thread consumer_a(consume);
  std::thread consumer_b(consume);
  std::thread producer([&] {
    for (std::uint64_t id = 1; id <= kFrames; ++id) {
      if (auto frame = pool.Acquire()) {
        frame->frame_id = id;
        AllocateFrame(*frame, PixelFormat::kNv12, 16, 16);
        slot.Publish(std::move(frame));
      }
      EXPECT_LE(pool.outstanding(), kCapacity);
    }
    producer_done = true;
  });
  producer.join();
  consumer_a.join();
  consumer_b.join();

  EXPECT_TRUE(order_ok.load());
  EXPECT_GT(taken.load(), 0u);
  const SlotStats stats = slot.stats();
  EXPECT_EQ(stats.published + pool.exhausted_count(), kFrames);
  slot.Clear();
  EXPECT_EQ(pool.outstanding(), 0u);
}

// Shutdown while consumers still hold frames: stop the producer, destroy the
// pool, then let the consumers release their frames.
TEST(FrameQueue, ShutdownWithOutstandingReferences) {
  auto pool = std::make_unique<FramePool>(3);
  LatestFrameSlot slot;
  std::atomic<bool> stop{false};
  std::vector<std::shared_ptr<const VideoFrame>> kept;
  std::thread producer([&] {
    std::uint64_t id = 0;
    while (!stop.load()) {
      if (auto frame = pool->Acquire()) {
        frame->frame_id = ++id;
        slot.Publish(std::move(frame));
      } else {
        std::this_thread::yield();
      }
    }
  });
  for (int i = 0; i < 2; ++i) {
    std::shared_ptr<const VideoFrame> frame;
    while (!(frame = slot.TakeIfNew(kept.empty() ? 0 : kept.back()->frame_id))) {
      std::this_thread::yield();
    }
    kept.push_back(std::move(frame));
  }
  stop = true;
  producer.join();
  slot.Clear();
  pool.reset();  // frames in `kept` now outlive their pool
  for (const auto& frame : kept) {
    EXPECT_GT(frame->frame_id, 0u);
  }
  kept.clear();
}

}  // namespace
}  // namespace relume
