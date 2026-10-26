// FrameSaver: opt-in local fixture capture. No camera needed; frames are
// generated with EncodeRgb8.

#include "relume/frame_saver.hpp"
#include "relume/image_io.hpp"
#include "relume/pixel_convert.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

namespace relume {
namespace {

namespace fs = std::filesystem;
using Clock = FrameSaver::Clock;
using std::chrono::milliseconds;

class TempDir {
 public:
  TempDir() {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    path_ = fs::temp_directory_path() / (std::string("relume_") + info->test_suite_name() + "_" + info->name());
    fs::remove_all(path_);
  }
  ~TempDir() {
    std::error_code ignored;
    fs::remove_all(path_, ignored);
  }
  std::string path() const { return path_.string(); }

 private:
  fs::path path_;
};

VideoFrame MakeFrame(std::uint64_t id, std::uint32_t width, std::uint32_t height) {
  RgbImage image = MakeOrientationPattern(width, height);
  image.pixels[0] = static_cast<std::uint8_t>(id);  // frames differ
  VideoFrame frame;
  EncodeRgb8(image.pixels.data(), width, height, PixelFormat::kNv12, YuvRange::kVideo, YuvMatrix::kBt709, frame);
  frame.frame_id = id;
  return frame;
}

bool WaitFor(const FrameSaver& saver, std::uint32_t written) {
  const auto deadline = Clock::now() + std::chrono::seconds(10);
  while (saver.written() + saver.failed() < written) {
    if (Clock::now() > deadline) {
      return false;
    }
    std::this_thread::sleep_for(milliseconds(1));
  }
  return true;
}

FrameSaverOptions Options(const TempDir& dir, std::uint32_t count, double interval_s, double delay_s) {
  FrameSaverOptions options;
  options.directory = dir.path();
  options.count = count;
  options.interval_s = interval_s;
  options.delay_s = delay_s;
  return options;
}

TEST(FrameSaver, ValidatesOptions) {
  FrameSaverOptions options;
  options.directory = "/tmp/x";
  options.count = 0;
  EXPECT_FALSE(FrameSaver::Validate(options).empty());
  options.count = FrameSaver::kMaxFrames + 1;
  EXPECT_FALSE(FrameSaver::Validate(options).empty());
  options.count = FrameSaver::kMaxFrames;
  EXPECT_TRUE(FrameSaver::Validate(options).empty());
  options.interval_s = -1.0;
  EXPECT_FALSE(FrameSaver::Validate(options).empty());
  options.interval_s = 0.0;
  options.directory.clear();
  EXPECT_FALSE(FrameSaver::Validate(options).empty());
}

// Each saved file is the exact ToRgb8 conversion of the offered frame.
TEST(FrameSaver, WritesExactRgbOfConsecutiveFrames) {
  TempDir dir;
  FrameSaver saver(Options(dir, 3, 0.0, 0.0));
  ASSERT_TRUE(saver.error().empty()) << saver.error();
  std::vector<VideoFrame> frames;
  const Clock::time_point t0 = Clock::now();
  for (std::uint64_t id = 1; id <= 5; ++id) {
    frames.push_back(MakeFrame(id, 64, 36));
    saver.Offer(frames.back(), t0 + milliseconds(33 * id));
    ASSERT_TRUE(WaitFor(saver, std::min<std::uint32_t>(static_cast<std::uint32_t>(id), 3)));
  }
  EXPECT_TRUE(saver.done());
  EXPECT_EQ(saver.written(), 3u);
  EXPECT_EQ(saver.skipped_busy(), 0u);
  const std::vector<std::string> paths = saver.paths();
  ASSERT_EQ(paths.size(), 3u);
  for (std::size_t i = 0; i < paths.size(); ++i) {
    std::string error;
    const std::optional<RgbImage> image = ReadPpm(paths[i], error);
    ASSERT_TRUE(image) << error;
    std::vector<std::uint8_t> expected;
    ASSERT_TRUE(ToRgb8(frames[i], expected));
    EXPECT_EQ(image->width, 64u);
    EXPECT_EQ(image->height, 36u);
    EXPECT_EQ(image->pixels, expected) << paths[i];
  }
  std::size_t files = 0;
  for ([[maybe_unused]] const auto& entry : fs::directory_iterator(dir.path())) {
    ++files;
  }
  EXPECT_EQ(files, 3u);  // never more than requested
}

TEST(FrameSaver, HonoursDelayAndInterval) {
  TempDir dir;
  FrameSaver saver(Options(dir, 2, 1.0, 5.0));
  const Clock::time_point t0 = Clock::now();
  auto at = [&](double s) { return t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(s)); };
  std::uint64_t id = 0;
  saver.Offer(MakeFrame(++id, 8, 8), at(0.0));  // starts the delay
  saver.Offer(MakeFrame(++id, 8, 8), at(4.9));
  ASSERT_TRUE(WaitFor(saver, 0));
  std::this_thread::sleep_for(milliseconds(20));
  EXPECT_EQ(saver.written(), 0u);
  saver.Offer(MakeFrame(++id, 8, 8), at(5.0));  // first save
  ASSERT_TRUE(WaitFor(saver, 1));
  saver.Offer(MakeFrame(++id, 8, 8), at(5.5));  // inside the interval
  std::this_thread::sleep_for(milliseconds(20));
  EXPECT_EQ(saver.written(), 1u);
  saver.Offer(MakeFrame(++id, 8, 8), at(6.0));  // second save
  ASSERT_TRUE(WaitFor(saver, 2));
  EXPECT_TRUE(saver.done());
  const std::vector<std::string> paths = saver.paths();
  ASSERT_EQ(paths.size(), 2u);
  EXPECT_NE(paths[0].find("_f3.ppm"), std::string::npos) << paths[0];
  EXPECT_NE(paths[1].find("_f5.ppm"), std::string::npos) << paths[1];
}

TEST(FrameSaver, SameFrameIsSavedOnce) {
  TempDir dir;
  FrameSaver saver(Options(dir, 2, 0.0, 0.0));
  const VideoFrame frame = MakeFrame(7, 8, 8);
  saver.Offer(frame, Clock::now());
  ASSERT_TRUE(WaitFor(saver, 1));
  saver.Offer(frame, Clock::now());
  std::this_thread::sleep_for(milliseconds(20));
  EXPECT_EQ(saver.written(), 1u);
  EXPECT_FALSE(saver.done());
}

// The caller never waits for the disk: with both buffers busy, a due frame is
// skipped and counted. 1920x1080 frames take milliseconds to convert and
// write, far longer than three back-to-back offers.
TEST(FrameSaver, SkipsInsteadOfBlockingWhenBuffersAreBusy) {
  TempDir dir;
  FrameSaver saver(Options(dir, 10, 0.0, 0.0));
  std::vector<VideoFrame> frames;
  for (std::uint64_t id = 1; id <= 3; ++id) {
    frames.push_back(MakeFrame(id, 1920, 1080));
  }
  const Clock::time_point start = Clock::now();
  for (const VideoFrame& frame : frames) {
    saver.Offer(frame, Clock::now());
  }
  const double offer_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  EXPECT_EQ(saver.skipped_busy(), 1u);
  EXPECT_LT(offer_ms, 50.0);  // three copies of 3 MB, no disk work
  ASSERT_TRUE(WaitFor(saver, 2));
  EXPECT_EQ(saver.written(), 2u);
}

TEST(FrameSaver, DestructorFinishesQueuedWrites) {
  TempDir dir;
  {
    FrameSaver saver(Options(dir, 2, 0.0, 0.0));
    saver.Offer(MakeFrame(1, 640, 480), Clock::now());
    saver.Offer(MakeFrame(2, 640, 480), Clock::now());
  }
  std::size_t files = 0;
  for ([[maybe_unused]] const auto& entry : fs::directory_iterator(dir.path())) {
    ++files;
  }
  EXPECT_EQ(files, 2u);
}

TEST(FrameSaver, UnusableDirectoryAcceptsNothing) {
  TempDir dir;
  fs::create_directories(dir.path());
  const std::string file = (fs::path(dir.path()) / "not_a_dir").string();
  std::ofstream(file) << "x";
  FrameSaverOptions bad = Options(dir, 2, 0.0, 0.0);
  bad.directory = (fs::path(file) / "sub").string();
  FrameSaver broken(bad);
  EXPECT_FALSE(broken.error().empty());
  EXPECT_TRUE(broken.done());
  broken.Offer(MakeFrame(1, 8, 8), Clock::now());
  EXPECT_EQ(broken.written(), 0u);
}

}  // namespace
}  // namespace relume
