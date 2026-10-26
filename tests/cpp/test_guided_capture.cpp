// GuidedCapture and FrameSaver's manual mode. No camera needed: frames are
// generated, and time is passed in explicitly.

#include "relume/guided_capture.hpp"
#include "relume/image_io.hpp"
#include "relume/pixel_convert.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <set>
#include <thread>

namespace relume {
namespace {

namespace fs = std::filesystem;
using Clock = GuidedCapture::Clock;
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
  fs::path operator/(const std::string& name) const { return path_ / name; }

 private:
  fs::path path_;
};

VideoFrame MakeFrame(std::uint64_t id, std::uint32_t width = 32, std::uint32_t height = 18) {
  RgbImage image = MakeOrientationPattern(width, height);
  image.pixels[0] = static_cast<std::uint8_t>(id);  // frames differ
  VideoFrame frame;
  EncodeRgb8(image.pixels.data(), width, height, PixelFormat::kBgra8, YuvRange::kVideo, YuvMatrix::kBt709, frame);
  frame.frame_id = id;
  return frame;
}

FrameSaverOptions Manual(const TempDir& dir) {
  FrameSaverOptions options;
  options.directory = dir.path();
  options.count = 0;
  return options;
}

std::vector<CaptureStep> SmallPlan(std::uint32_t burst_frames = 4) {
  return {{CaptureStep::Kind::kStill, "still_01", "A", "do A", 1},
          {CaptureStep::Kind::kStill, "still_02", "B", "do B", 1},
          {CaptureStep::Kind::kBurst, "burst_a", "Burst", "stay still", burst_frames}};
}

Clock::time_point After(Clock::time_point t, double s) {
  return t + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(s));
}

// Calls Update until the current shot is no longer busy.
bool Settle(GuidedCapture& capture, Clock::time_point now) {
  const auto deadline = Clock::now() + std::chrono::seconds(10);
  while (capture.busy()) {
    if (Clock::now() > deadline) {
      return false;
    }
    std::this_thread::sleep_for(milliseconds(1));
    capture.Update(now);
  }
  return true;
}

std::uint8_t FirstByte(const fs::path& path) {
  std::string error;
  const std::optional<RgbImage> image = ReadPpm(path.string(), error);
  return image ? image->pixels[0] : 0;
}

std::size_t CountFiles(const fs::path& dir) {
  std::size_t n = 0;
  for (const auto& entry : fs::recursive_directory_iterator(dir)) {
    n += entry.is_regular_file() ? 1 : 0;
  }
  return n;
}

TEST(GuidedCapture, DefaultPlanHasTwelveStillsAndTwoBursts) {
  const std::vector<CaptureStep> plan = DefaultCapturePlan();
  std::set<std::string> names;
  int stills = 0, bursts = 0;
  for (const CaptureStep& step : plan) {
    EXPECT_FALSE(step.instruction.empty());
    EXPECT_FALSE(step.title.empty());
    EXPECT_TRUE(names.insert(step.name).second) << step.name;
    if (step.kind == CaptureStep::Kind::kStill) {
      ++stills;
      EXPECT_EQ(step.frames, 1u);
    } else {
      ++bursts;
      EXPECT_EQ(step.frames, 20u);
    }
  }
  EXPECT_EQ(stills, 12);
  EXPECT_EQ(bursts, 2);
}

TEST(GuidedCapture, StillWaitsForCountdownThenSavesOneFrame) {
  TempDir dir;
  FrameSaver saver(Manual(dir));
  GuidedCapture capture(saver, SmallPlan());
  const Clock::time_point t0 = Clock::now();
  EXPECT_DOUBLE_EQ(capture.countdown_remaining(t0), 0.0);
  capture.StartCapture(t0);
  EXPECT_EQ(capture.status(0), ShotStatus::kCountdown);
  EXPECT_NEAR(capture.countdown_remaining(After(t0, 1.0)), 2.0, 1e-6);

  capture.Update(After(t0, 2.9));
  capture.OnFrame(MakeFrame(1), After(t0, 2.9));  // still counting down: ignored
  EXPECT_EQ(capture.status(0), ShotStatus::kCountdown);
  capture.Update(After(t0, 3.0));
  EXPECT_EQ(capture.status(0), ShotStatus::kCapturing);
  capture.OnFrame(MakeFrame(2), After(t0, 3.0));
  capture.OnFrame(MakeFrame(3), After(t0, 3.03));  // only one frame per still
  EXPECT_EQ(capture.status(0), ShotStatus::kWriting);
  EXPECT_EQ(capture.saved_stills(), 0u);  // not counted before the writer confirms
  ASSERT_TRUE(Settle(capture, After(t0, 3.1)));
  EXPECT_EQ(capture.status(0), ShotStatus::kSaved);
  EXPECT_EQ(capture.saved_stills(), 1u);
  EXPECT_EQ(capture.files_written(), 1u);
  EXPECT_EQ(FirstByte(dir / "still_01.ppm"), 2);
  EXPECT_EQ(CountFiles(dir.path()), 1u);
}

TEST(GuidedCapture, RetakeReplacesTheShotWithoutAddingFiles) {
  TempDir dir;
  FrameSaver saver(Manual(dir));
  GuidedCapture capture(saver, SmallPlan());
  Clock::time_point t = Clock::now();
  std::uint64_t id = 0;
  for (int take = 0; take < 2; ++take) {
    capture.StartCapture(t);
    t = After(t, 3.0);
    capture.Update(t);
    capture.OnFrame(MakeFrame(++id), t);
    ASSERT_TRUE(Settle(capture, t));
  }
  EXPECT_EQ(capture.status(0), ShotStatus::kSaved);
  EXPECT_EQ(capture.saved_stills(), 1u);  // still one shot
  EXPECT_EQ(FirstByte(dir / "still_01.ppm"), 2);  // the retake
  EXPECT_EQ(CountFiles(dir.path()), 1u);
}

TEST(GuidedCapture, BusyShotCannotBeChangedOrRestarted) {
  TempDir dir;
  FrameSaver saver(Manual(dir));
  GuidedCapture capture(saver, SmallPlan());
  const Clock::time_point t0 = Clock::now();
  capture.StartCapture(t0);
  capture.Select(1);
  EXPECT_EQ(capture.current(), 0u);
  capture.StartCapture(After(t0, 1.0));  // does not restart the countdown
  EXPECT_NEAR(capture.countdown_remaining(After(t0, 1.0)), 2.0, 1e-6);
  capture.Update(After(t0, 3.0));
  capture.OnFrame(MakeFrame(1), After(t0, 3.0));
  ASSERT_TRUE(Settle(capture, After(t0, 3.0)));
  capture.Select(1);
  EXPECT_EQ(capture.current(), 1u);
}

TEST(GuidedCapture, BurstSavesConsecutiveFramesAsOneFolder) {
  TempDir dir;
  FrameSaver saver(Manual(dir));
  GuidedCapture capture(saver, SmallPlan(4));
  capture.Select(2);
  Clock::time_point t = Clock::now();
  capture.StartCapture(t);
  t = After(t, 3.0);
  capture.Update(t);
  ASSERT_EQ(capture.status(2), ShotStatus::kCapturing);
  for (std::uint64_t id = 10; id < 14; ++id) {
    capture.OnFrame(MakeFrame(id), t);
    EXPECT_EQ(capture.burst_progress(), id - 9);
    // Give the writer time between frames, like a 30 fps camera does.
    for (int i = 0; i < 1000 && saver.written() < id - 9; ++i) {
      std::this_thread::sleep_for(milliseconds(1));
    }
  }
  ASSERT_TRUE(Settle(capture, t));
  EXPECT_EQ(capture.status(2), ShotStatus::kSaved) << capture.message(2);
  EXPECT_EQ(capture.saved_bursts(), 1u);
  EXPECT_EQ(capture.files_written(), 4u);
  EXPECT_FALSE(fs::exists(dir / "burst_a.partial"));
  ASSERT_TRUE(fs::is_directory(dir / "burst_a"));
  EXPECT_EQ(CountFiles(dir / "burst_a"), 4u);
  for (int i = 1; i <= 4; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%02d.ppm", i);
    EXPECT_EQ(FirstByte(dir / "burst_a" / name), 9 + i);
  }
}

// Frames arriving faster than the writer can save them: a refused frame
// breaks the consecutive run, so the burst keeps nothing and is not counted.
TEST(GuidedCapture, BurstWithASkippedFrameKeepsNothing) {
  TempDir dir;
  FrameSaver saver(Manual(dir));
  GuidedCapture capture(saver, SmallPlan(6));
  capture.Select(2);
  Clock::time_point t = Clock::now();
  capture.StartCapture(t);
  t = After(t, 3.0);
  capture.Update(t);
  std::vector<VideoFrame> frames;
  for (std::uint64_t id = 1; id <= 6; ++id) {
    frames.push_back(MakeFrame(id, 1920, 1080));
  }
  for (const VideoFrame& frame : frames) {
    capture.OnFrame(frame, t);  // back to back: the third finds both buffers busy
  }
  ASSERT_TRUE(Settle(capture, t));
  EXPECT_EQ(capture.status(2), ShotStatus::kFailed);
  EXPECT_NE(capture.message(2).find("skipped"), std::string::npos) << capture.message(2);
  EXPECT_EQ(capture.saved_bursts(), 0u);
  EXPECT_EQ(capture.files_written(), 0u);
  EXPECT_GE(saver.skipped_busy(), 1u);
  EXPECT_FALSE(fs::exists(dir / "burst_a"));
  EXPECT_FALSE(fs::exists(dir / "burst_a.partial"));
}

// A failed write is reported as a failure and leaves no file that looks
// like a current shot (here a folder occupies the destination name).
TEST(GuidedCapture, FailedStillWriteIsNotCounted) {
  TempDir dir;
  fs::create_directories(dir / "still_01.ppm");
  FrameSaver saver(Manual(dir));
  GuidedCapture capture(saver, SmallPlan());
  ASSERT_EQ(capture.status(0), ShotStatus::kSaved);  // something exists from "an earlier session"
  Clock::time_point t = Clock::now();
  capture.StartCapture(t);
  t = After(t, 3.0);
  capture.Update(t);
  capture.OnFrame(MakeFrame(1), t);
  ASSERT_TRUE(Settle(capture, t));
  EXPECT_EQ(capture.status(0), ShotStatus::kFailed);
  EXPECT_NE(capture.message(0).find("Not saved"), std::string::npos);
  EXPECT_EQ(capture.saved_stills(), 0u);
  EXPECT_EQ(capture.files_written(), 0u);
  EXPECT_FALSE(fs::exists(dir / "still_01.ppm.partial"));
}

TEST(GuidedCapture, FailedRetakeRemovesThePreviousShot) {
  TempDir dir;
  FrameSaver saver(Manual(dir));
  GuidedCapture capture(saver, SmallPlan());
  Clock::time_point t = Clock::now();
  capture.StartCapture(t);
  t = After(t, 3.0);
  capture.Update(t);
  capture.OnFrame(MakeFrame(1), t);
  ASSERT_TRUE(Settle(capture, t));
  ASSERT_TRUE(fs::exists(dir / "still_01.ppm"));
  // Retake with a frame whose layout is invalid: the writer cannot convert it.
  VideoFrame bad = MakeFrame(2);
  bad.data.resize(4);
  capture.StartCapture(t);
  t = After(t, 3.0);
  capture.Update(t);
  capture.OnFrame(bad, t);
  ASSERT_TRUE(Settle(capture, t));
  EXPECT_EQ(capture.status(0), ShotStatus::kFailed);
  EXPECT_FALSE(fs::exists(dir / "still_01.ppm"));
  EXPECT_EQ(capture.saved_stills(), 0u);
}

TEST(GuidedCapture, RecognizesShotsFromAnEarlierSession) {
  TempDir dir;
  fs::create_directories(dir / "burst_a");
  std::ofstream(dir / "still_02.ppm") << "x";
  FrameSaver saver(Manual(dir));
  GuidedCapture capture(saver, SmallPlan());
  EXPECT_EQ(capture.status(0), ShotStatus::kNotTaken);
  EXPECT_EQ(capture.status(1), ShotStatus::kSaved);
  EXPECT_EQ(capture.status(2), ShotStatus::kSaved);
  EXPECT_EQ(capture.files_written(), 0u);  // nothing written in this session
}

TEST(FrameSaverManual, RejectsPathsOutsideTheDirectory) {
  TempDir dir;
  FrameSaver saver(Manual(dir));
  const VideoFrame frame = MakeFrame(1);
  EXPECT_EQ(saver.Enqueue(frame, "../escape.ppm"), 0u);
  EXPECT_EQ(saver.Enqueue(frame, "/tmp/escape.ppm"), 0u);
  EXPECT_EQ(saver.EnqueueRemoveDir(".."), 0u);
  EXPECT_EQ(saver.EnqueueCommitDir("a", "../b", 1), 0u);
  EXPECT_NE(saver.Enqueue(frame, "sub/ok.ppm"), 0u);
}

TEST(FrameSaverManual, OfferDoesNothingInManualMode) {
  TempDir dir;
  FrameSaver saver(Manual(dir));
  saver.Offer(MakeFrame(1), Clock::now());
  std::this_thread::sleep_for(milliseconds(20));
  EXPECT_EQ(saver.written(), 0u);
  EXPECT_FALSE(saver.done());  // manual mode never "finishes"
}

}  // namespace
}  // namespace relume
