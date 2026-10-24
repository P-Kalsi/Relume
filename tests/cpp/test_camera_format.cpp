#include "relume/camera.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace relume {
namespace {

CameraFormat Format(std::uint32_t w, std::uint32_t h, std::vector<FrameRateRange> rates, std::string code = "420v") {
  return {w, h, std::move(rates), std::move(code)};
}

const CaptureRequest kDefault{};  // 1280x720 @ 30, NV12

TEST(SelectFormat, ExactSizeAt30Wins) {
  const std::vector<CameraFormat> formats = {
      Format(1920, 1080, {{1, 30}}), Format(1280, 720, {{1, 30}}), Format(640, 480, {{1, 30}})};
  const auto choice = SelectFormat(formats, kDefault);
  ASSERT_TRUE(choice);
  EXPECT_EQ(choice->index, 1u);
  EXPECT_DOUBLE_EQ(choice->fps, 30.0);
}

TEST(SelectFormat, NearestSameAspectWhenExactSizeMissing) {
  // Like the Phase 0 camera: no 720p. 16:9 1080p beats 4:3 formats.
  const std::vector<CameraFormat> formats = {
      Format(640, 480, {{1, 30}}), Format(1920, 1080, {{1, 30}}), Format(1280, 960, {{1, 30}})};
  const auto choice = SelectFormat(formats, kDefault);
  ASSERT_TRUE(choice);
  EXPECT_EQ(choice->index, 1u);
}

TEST(SelectFormat, SmallerSixteenNineIsPreferredOverMuchLarger) {
  const std::vector<CameraFormat> formats = {Format(3840, 2160, {{1, 30}}), Format(960, 540, {{1, 30}})};
  const auto choice = SelectFormat(formats, kDefault);
  ASSERT_TRUE(choice);
  // |log(960*540 / 1280*720)| = 0.575 < |log(3840*2160 / 1280*720)| = 2.197
  EXPECT_EQ(choice->index, 1u);
}

TEST(SelectFormat, FourThreeOnlyDevicePicksNearestArea) {
  const std::vector<CameraFormat> formats = {
      Format(320, 240, {{1, 30}}), Format(1024, 768, {{1, 30}}), Format(2592, 1944, {{1, 30}})};
  const auto choice = SelectFormat(formats, kDefault);
  ASSERT_TRUE(choice);
  EXPECT_EQ(choice->index, 1u);
}

TEST(SelectFormat, FrameRateBeatsExactSize) {
  // 720p only at 60 fixed versus 1080p that can do 30.
  const std::vector<CameraFormat> formats = {Format(1280, 720, {{60, 60}}), Format(1920, 1080, {{15, 30}})};
  const auto choice = SelectFormat(formats, kDefault);
  ASSERT_TRUE(choice);
  EXPECT_EQ(choice->index, 1u);
  EXPECT_DOUBLE_EQ(choice->fps, 30.0);
}

TEST(SelectFormat, RateIsClampedIntoTheRange) {
  const std::vector<CameraFormat> formats = {Format(1280, 720, {{5, 24}})};
  const auto choice = SelectFormat(formats, kDefault);
  ASSERT_TRUE(choice);
  EXPECT_DOUBLE_EQ(choice->fps, 24.0);
}

TEST(SelectFormat, BestRangeWithinAFormatIsUsed) {
  const std::vector<CameraFormat> formats = {Format(1280, 720, {{60, 60}, {1, 30}})};
  const auto choice = SelectFormat(formats, kDefault);
  ASSERT_TRUE(choice);
  EXPECT_DOUBLE_EQ(choice->fps, 30.0);
}

TEST(SelectFormat, UnusableInputsReturnNothing) {
  EXPECT_FALSE(SelectFormat({}, kDefault));
  const std::vector<CameraFormat> broken = {Format(0, 720, {{1, 30}}), Format(1280, 720, {}),
                                            Format(1280, 720, {{30, 1}})};
  EXPECT_FALSE(SelectFormat(broken, kDefault));
  CaptureRequest bad = kDefault;
  bad.fps = 0;
  EXPECT_FALSE(SelectFormat(std::vector<CameraFormat>{Format(1280, 720, {{1, 30}})}, bad));
}

TEST(ChooseOutputPixelFormat, PrefersNativeNv12) {
  const std::vector<PixelFormatOption> options = {{PixelFormat::kBgra8, YuvRange::kFull, "BGRA"},
                                                  {PixelFormat::kNv12, YuvRange::kVideo, "420v"},
                                                  {PixelFormat::kNv12, YuvRange::kFull, "420f"}};
  EXPECT_EQ(ChooseOutputPixelFormat(options, PixelFormat::kNv12, "420f"), 2u);
  EXPECT_EQ(ChooseOutputPixelFormat(options, PixelFormat::kNv12, "420v"), 1u);
  // Native format is not YCbCr 4:2:0: video range first.
  EXPECT_EQ(ChooseOutputPixelFormat(options, PixelFormat::kNv12, "yuvs"), 1u);
  EXPECT_EQ(ChooseOutputPixelFormat(options, PixelFormat::kBgra8, "420v"), 0u);
}

TEST(ChooseOutputPixelFormat, FallsBackToTheOtherKind) {
  const std::vector<PixelFormatOption> bgra_only = {{PixelFormat::kBgra8, YuvRange::kFull, "BGRA"}};
  EXPECT_EQ(ChooseOutputPixelFormat(bgra_only, PixelFormat::kNv12, "420v"), 0u);
  const std::vector<PixelFormatOption> full_only = {{PixelFormat::kNv12, YuvRange::kFull, "420f"}};
  EXPECT_EQ(ChooseOutputPixelFormat(full_only, PixelFormat::kBgra8, "420v"), 0u);
  EXPECT_FALSE(ChooseOutputPixelFormat({}, PixelFormat::kNv12, "420v"));
}

TEST(FallbackYuvMatrix, SdIs601AndHdIs709) {
  EXPECT_EQ(FallbackYuvMatrix(640, 480), YuvMatrix::kBt601);
  EXPECT_EQ(FallbackYuvMatrix(960, 540), YuvMatrix::kBt601);
  EXPECT_EQ(FallbackYuvMatrix(1280, 720), YuvMatrix::kBt709);
  EXPECT_EQ(FallbackYuvMatrix(1920, 1080), YuvMatrix::kBt709);
}

TEST(VirtualCamera, ObsIsRecognisedCaseInsensitively) {
  EXPECT_TRUE(IsLikelyVirtualCamera("OBS Virtual Camera"));
  EXPECT_TRUE(IsLikelyVirtualCamera("obs virtual camera (2)"));
  EXPECT_FALSE(IsLikelyVirtualCamera("FaceTime HD Camera"));
  EXPECT_FALSE(IsLikelyVirtualCamera("OBS"));
  EXPECT_FALSE(IsLikelyVirtualCamera(""));
}

CameraDeviceInfo Device(std::string id, std::string name) {
  CameraDeviceInfo info;
  info.id = std::move(id);
  info.is_virtual = IsLikelyVirtualCamera(name);
  info.name = std::move(name);
  return info;
}

TEST(ChooseDefaultDevice, NeverPicksVirtualCamera) {
  const std::vector<CameraDeviceInfo> obs_first = {Device("obs", "OBS Virtual Camera"),
                                                   Device("cam", "FaceTime HD Camera")};
  EXPECT_EQ(ChooseDefaultDevice(obs_first, ""), 1u);
  EXPECT_EQ(ChooseDefaultDevice(obs_first, "obs"), 1u) << "even if remembered";
  const std::vector<CameraDeviceInfo> obs_only = {Device("obs", "OBS Virtual Camera")};
  EXPECT_FALSE(ChooseDefaultDevice(obs_only, ""));
  EXPECT_FALSE(ChooseDefaultDevice({}, ""));
}

TEST(ChooseDefaultDevice, PrefersRememberedDevice) {
  const std::vector<CameraDeviceInfo> devices = {Device("a", "FaceTime HD Camera"), Device("b", "iPhone Camera")};
  EXPECT_EQ(ChooseDefaultDevice(devices, "b"), 1u);
  EXPECT_EQ(ChooseDefaultDevice(devices, "gone"), 0u);
}

}  // namespace
}  // namespace relume
