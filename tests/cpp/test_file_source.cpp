// FileSource, PPM input, event queue and latency histogram. No camera needed.

#include "relume/event_queue.hpp"
#include "relume/file_source.hpp"
#include "relume/frame_queue.hpp"
#include "relume/image_io.hpp"
#include "relume/pixel_convert.hpp"
#include "relume/stats.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>

namespace relume {
namespace {

namespace fs = std::filesystem;

// A per-test temporary directory, removed afterwards.
class TempDir {
 public:
  TempDir() {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    path_ = fs::temp_directory_path() / (std::string("relume_") + info->test_suite_name() + "_" + info->name());
    fs::remove_all(path_);
    fs::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ignored;
    fs::remove_all(path_, ignored);
  }
  std::string File(const char* name) const { return (path_ / name).string(); }

 private:
  fs::path path_;
};

// Waits (bounded) until `predicate` holds.
template <typename Predicate>
bool WaitFor(Predicate predicate, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() > deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

TEST(Ppm, RoundTrip) {
  TempDir dir;
  const RgbImage image = MakeOrientationPattern(33, 17);
  ASSERT_TRUE(WritePpm(dir.File("a.ppm"), image));
  std::string error;
  const auto loaded = ReadPpm(dir.File("a.ppm"), error);
  ASSERT_TRUE(loaded) << error;
  EXPECT_EQ(loaded->width, 33u);
  EXPECT_EQ(loaded->height, 17u);
  EXPECT_EQ(loaded->pixels, image.pixels);
}

TEST(Ppm, HeaderCommentsAreSkipped) {
  TempDir dir;
  {
    std::ofstream out(dir.File("c.ppm"), std::ios::binary);
    out << "P6\n# comment\n2 1\n# another\n255\n";
    const char pixels[] = {1, 2, 3, 4, 5, 6};
    out.write(pixels, sizeof(pixels));
  }
  std::string error;
  const auto loaded = ReadPpm(dir.File("c.ppm"), error);
  ASSERT_TRUE(loaded) << error;
  EXPECT_EQ(loaded->pixels, (std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6}));
}

TEST(Ppm, MalformedFilesFailCleanly) {
  TempDir dir;
  auto write = [&](const char* name, const std::string& bytes) {
    std::ofstream out(dir.File(name), std::ios::binary);
    out << bytes;
  };
  write("ascii.ppm", "P3\n1 1\n255\n0 0 0\n");
  write("truncated.ppm", "P6\n4 4\n255\nabc");
  write("maxval.ppm", "P6\n1 1\n65535\n\x01\x02\x03\x04\x05\x06");
  write("huge.ppm", "P6\n999999 999999\n255\n");
  write("garbage.ppm", "P6\nx y\n255\n");
  write("empty.ppm", "");
  for (const char* name : {"ascii.ppm", "truncated.ppm", "maxval.ppm", "huge.ppm", "garbage.ppm", "empty.ppm",
                           "missing.ppm"}) {
    std::string error;
    EXPECT_FALSE(ReadPpm(dir.File(name), error)) << name;
    EXPECT_FALSE(error.empty()) << name;
  }
}

TEST(OrientationPattern, CornersAndMarker) {
  const RgbImage image = MakeOrientationPattern(64, 32);
  auto at = [&](std::uint32_t x, std::uint32_t y) { return &image.pixels[(std::size_t{y} * 64 + x) * 3]; };
  EXPECT_EQ(at(0, 0)[0], pattern::kTopLeft[0]);
  EXPECT_EQ(at(63, 0)[1], pattern::kTopRight[1]);
  EXPECT_EQ(at(0, 31)[2], pattern::kBottomLeft[2]);
  EXPECT_EQ(at(63, 31)[0], pattern::kBottomRight[0]);
  EXPECT_EQ(at(5, 3)[0], pattern::kMarker[0]);  // marker spans x 4..11, y 2..5
  EXPECT_EQ(at(5, 3)[1], pattern::kMarker[1]);
}

TEST(FileSource, DeterministicIdsAndTimestamps) {
  CameraEventQueue events;
  LatestFrameSlot slot;
  FileSourceOptions options;
  options.pattern_width = 64;
  options.pattern_height = 36;
  options.fps = 30.0;
  options.max_frames = 50;
  options.realtime = false;
  FileSource source(options, slot, events);
  source.Start();
  std::uint64_t last = 0;
  // Collect whatever frames we observe until the source ends.
  ASSERT_TRUE(WaitFor([&] {
    if (auto frame = slot.TakeIfNew(last)) {
      EXPECT_GT(frame->frame_id, last);
      EXPECT_EQ(frame->capture_time_ns,
                static_cast<std::int64_t>(static_cast<double>(frame->frame_id - 1) * 1e9 / 30.0));
      EXPECT_EQ(frame->clock, TimestampClock::kSynthetic);
      EXPECT_EQ(frame->width, 64u);
      EXPECT_EQ(frame->height, 36u);
      last = frame->frame_id;
    }
    return source.state() == SourceState::kIdle;
  }));
  source.Stop();
  const auto final_frame = slot.Latest();
  ASSERT_TRUE(final_frame);
  EXPECT_EQ(final_frame->frame_id, 50u);
  const CaptureStats stats = source.stats();
  EXPECT_EQ(stats.frames_received, 50u);
  EXPECT_EQ(stats.frames_published + stats.dropped_pool, 50u);

  bool started = false;
  bool ended = false;
  while (auto event = events.Poll()) {
    started |= event->kind == CameraEvent::Kind::kSourceStarted;
    ended |= event->kind == CameraEvent::Kind::kSourceEnded;
  }
  EXPECT_TRUE(started);
  EXPECT_TRUE(ended);
}

TEST(FileSource, EncodesEveryPixelFormat) {
  for (PixelFormat format : {PixelFormat::kBgra8, PixelFormat::kRgb8, PixelFormat::kNv12}) {
    CameraEventQueue events;
    LatestFrameSlot slot;
    FileSourceOptions options;
    options.pattern_width = 32;
    options.pattern_height = 16;
    options.format = format;
    options.range = YuvRange::kFull;
    options.matrix = YuvMatrix::kBt601;
    options.max_frames = 1;
    options.realtime = false;
    FileSource source(options, slot, events);
    source.Start();
    ASSERT_TRUE(WaitFor([&] { return source.state() == SourceState::kIdle; })) << ToString(format);
    const auto frame = slot.Latest();
    ASSERT_TRUE(frame);
    EXPECT_EQ(frame->format, format);
    EXPECT_EQ(ValidateLayout(*frame), "");
    std::vector<std::uint8_t> rgb;
    ASSERT_TRUE(ToRgb8(*frame, rgb));
    // Top-left pixel is red within the NV12 round-trip tolerance (2).
    EXPECT_NEAR(rgb[0], pattern::kTopLeft[0], 2) << ToString(format);
    EXPECT_NEAR(rgb[1], pattern::kTopLeft[1], 2) << ToString(format);
    EXPECT_NEAR(rgb[2], pattern::kTopLeft[2], 2) << ToString(format);
  }
}

TEST(FileSource, ReadsPpmFile) {
  TempDir dir;
  const RgbImage image = MakeOrientationPattern(20, 10);
  ASSERT_TRUE(WritePpm(dir.File("in.ppm"), image));
  CameraEventQueue events;
  LatestFrameSlot slot;
  FileSourceOptions options;
  options.path = dir.File("in.ppm");
  options.format = PixelFormat::kRgb8;
  options.max_frames = 1;
  options.realtime = false;
  FileSource source(options, slot, events);
  source.Start();
  ASSERT_TRUE(WaitFor([&] { return source.state() == SourceState::kIdle; }));
  std::vector<std::uint8_t> rgb;
  ASSERT_TRUE(ToRgb8(*slot.Latest(), rgb));
  EXPECT_EQ(rgb, image.pixels);
}

TEST(FileSource, MissingOrMalformedFileFailsWithoutCrashing) {
  TempDir dir;
  {
    std::ofstream out(dir.File("bad.ppm"), std::ios::binary);
    out << "not an image";
  }
  for (const std::string& path : {dir.File("bad.ppm"), dir.File("missing.ppm")}) {
    CameraEventQueue events;
    LatestFrameSlot slot;
    FileSourceOptions options;
    options.path = path;
    FileSource source(options, slot, events);
    source.Start();
    EXPECT_EQ(source.state(), SourceState::kFailed);
    const auto event = events.Poll();
    ASSERT_TRUE(event);
    EXPECT_EQ(event->kind, CameraEvent::Kind::kSourceFailed);
    EXPECT_FALSE(event->message.empty());
    EXPECT_FALSE(slot.Latest());
    source.Stop();
    EXPECT_EQ(source.state(), SourceState::kFailed);
  }
}

TEST(FileSource, RealtimePacingAndPromptStop) {
  CameraEventQueue events;
  LatestFrameSlot slot;
  FileSourceOptions options;
  options.pattern_width = 16;
  options.pattern_height = 16;
  options.fps = 2.0;  // 500 ms between frames
  FileSource source(options, slot, events);
  source.Start();
  ASSERT_TRUE(WaitFor([&] { return source.state() == SourceState::kRunning; }));
  const auto stop_start = std::chrono::steady_clock::now();
  source.Stop();  // must interrupt the 500 ms wait, not sleep through it
  const auto stop_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stop_start).count();
  EXPECT_LT(stop_ms, 200.0);
  EXPECT_EQ(source.state(), SourceState::kIdle);
  EXPECT_LE(source.stats().frames_received, 2u);
}

// Rapid start/stop and switching between sources with a consumer running, as
// the app does when the user changes cameras. Run under asan and tsan too.
TEST(FileSource, RapidStartStopAndSwitching) {
  CameraEventQueue events(1024);
  LatestFrameSlot slot;
  std::atomic<bool> done{false};
  std::thread consumer([&] {
    std::uint64_t last = 0;
    std::shared_ptr<const VideoFrame> held;
    while (!done.load()) {
      if (auto frame = slot.TakeIfNew(last)) {
        last = frame->frame_id;
        held = std::move(frame);
      }
      std::this_thread::yield();
    }
  });
  for (int cycle = 0; cycle < 100; ++cycle) {
    FileSourceOptions options;
    options.pattern_width = 64;
    options.pattern_height = 32;
    options.format = cycle % 2 == 0 ? PixelFormat::kNv12 : PixelFormat::kBgra8;
    options.realtime = cycle % 3 != 0;
    options.fps = 240.0;
    auto source = std::make_unique<FileSource>(options, slot, events);
    source->Start();
    source->Start();  // idempotent
    if (cycle % 4 == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    source->Stop();
    source->Stop();  // idempotent
    EXPECT_EQ(source->state(), SourceState::kIdle);
    if (cycle % 5 == 0) {
      slot.Clear();
    }
    source.reset();  // frames still held by `slot` or the consumer outlive the source's pool
  }
  done = true;
  consumer.join();
  slot.Clear();
}

TEST(FileSource, DestructorStopsRunningSource) {
  CameraEventQueue events;
  LatestFrameSlot slot;
  {
    FileSourceOptions options;
    options.pattern_width = 16;
    options.pattern_height = 16;
    options.realtime = false;
    FileSource source(options, slot, events);
    source.Start();
    ASSERT_TRUE(WaitFor([&] { return source.state() == SourceState::kRunning; }));
  }
  EXPECT_TRUE(slot.Latest());
}

TEST(EventQueue, BoundedAndOrdered) {
  CameraEventQueue queue(3);
  for (int i = 0; i < 5; ++i) {
    queue.Post({CameraEvent::Kind::kWarning, "", std::to_string(i)});
  }
  EXPECT_EQ(queue.discarded(), 2u);
  EXPECT_EQ(queue.Poll()->message, "2");
  EXPECT_EQ(queue.Poll()->message, "3");
  EXPECT_EQ(queue.Poll()->message, "4");
  EXPECT_FALSE(queue.Poll());
}

TEST(LatencyHistogram, Percentiles) {
  LatencyHistogram histogram;
  EXPECT_EQ(histogram.Percentile(0.5), 0.0);
  for (int i = 1; i <= 100; ++i) {
    histogram.Record(i * 0.1);  // 0.1 .. 10 ms
  }
  EXPECT_EQ(histogram.count(), 100u);
  EXPECT_NEAR(histogram.Percentile(0.5), 5.0, LatencyHistogram::kBucketMs + 1e-9);
  EXPECT_NEAR(histogram.Percentile(0.95), 9.5, LatencyHistogram::kBucketMs + 1e-9);
  histogram.Record(1e9);    // clamps to the last bucket
  histogram.Record(-1.0);   // and the first
  EXPECT_EQ(histogram.count(), 102u);
  histogram.Reset();
  EXPECT_EQ(histogram.count(), 0u);
}

}  // namespace
}  // namespace relume
