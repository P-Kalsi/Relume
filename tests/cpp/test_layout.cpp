#include "relume/layout.hpp"

#include <gtest/gtest.h>

namespace relume {
namespace {

void ExpectRect(PixelRect r, std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h) {
  EXPECT_EQ(r.x, x);
  EXPECT_EQ(r.y, y);
  EXPECT_EQ(r.width, w);
  EXPECT_EQ(r.height, h);
}

TEST(FitRect, SameAspectFillsExactly) {
  ExpectRect(FitRect(1280, 720, 1280, 720), 0, 0, 1280, 720);
  ExpectRect(FitRect(1920, 1080, 1280, 720), 0, 0, 1280, 720);
  // Retina: a 1280x720 output in a 2560x1440 framebuffer.
  ExpectRect(FitRect(1280, 720, 2560, 1440), 0, 0, 2560, 1440);
}

TEST(FitRect, FourThreeIntoSixteenNineIsPillarboxed) {
  ExpectRect(FitRect(640, 480, 1280, 720), 160, 0, 960, 720);
}

TEST(FitRect, WideIntoTallIsLetterboxed) {
  ExpectRect(FitRect(1280, 720, 800, 1000), 0, 275, 800, 450);
}

TEST(FitRect, OddFramebufferSizesStayCentredAndUnstretched) {
  // Window dragged to an arbitrary size on a 2x display.
  const PixelRect r = FitRect(1280, 720, 2001, 1333);
  EXPECT_EQ(r.width, 2001u);
  EXPECT_EQ(r.height, 1126u);  // 2001 * 9 / 16 = 1125.56 -> 1126
  EXPECT_EQ(r.y, (1333u - 1126u) / 2);
  // Aspect error is below one pixel.
  EXPECT_NEAR(static_cast<double>(r.width) / r.height, 16.0 / 9.0, 2.0 / r.height);
}

TEST(FitRect, DegenerateSizes) {
  ExpectRect(FitRect(0, 720, 1280, 720), 0, 0, 0, 0);
  ExpectRect(FitRect(1280, 720, 0, 0), 0, 0, 0, 0);
  ExpectRect(FitRect(1, 1, 1, 1), 0, 0, 1, 1);
  // Extremely wide source still yields at least one row.
  ExpectRect(FitRect(10000, 1, 10, 10), 0, 4, 10, 1);
}

}  // namespace
}  // namespace relume
