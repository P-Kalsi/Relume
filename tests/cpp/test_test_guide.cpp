#include "relume/test_guide.hpp"

#include <gtest/gtest.h>

namespace relume {
namespace {

using Clock = TestGuide::Clock;

Clock::time_point At(Clock::time_point start, double seconds) {
  return start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}

std::vector<GuideStep> ThreeSteps() { return {{"a", "", 10.0}, {"b", "", 20.0}, {"c", "", 0.0}}; }

TEST(TestGuide, FollowsTheClock) {
  const Clock::time_point start = Clock::now();
  TestGuide guide(ThreeSteps(), start);
  EXPECT_EQ(guide.current(), 0u);
  EXPECT_DOUBLE_EQ(guide.step_start(1), 10.0);
  EXPECT_DOUBLE_EQ(guide.step_start(2), 30.0);
  EXPECT_FALSE(guide.Update(At(start, 9.9)));
  EXPECT_NEAR(guide.remaining(At(start, 9.0)), 1.0, 1e-6);
  EXPECT_TRUE(guide.Update(At(start, 10.0)));
  EXPECT_EQ(guide.current(), 1u);
  EXPECT_TRUE(guide.Update(At(start, 100.0)));
  EXPECT_EQ(guide.current(), 2u);
  EXPECT_TRUE(guide.finished());
  EXPECT_EQ(guide.remaining(At(start, 200.0)), 0.0);
  EXPECT_FALSE(guide.Update(At(start, 500.0))) << "the last step is open-ended";
}

TEST(TestGuide, SkipShiftsTheRestEarlier) {
  const Clock::time_point start = Clock::now();
  TestGuide guide(ThreeSteps(), start);
  guide.Skip(At(start, 4.0));
  EXPECT_EQ(guide.current(), 1u);
  EXPECT_DOUBLE_EQ(guide.step_start(1), 4.0);
  EXPECT_DOUBLE_EQ(guide.step_start(2), 24.0) << "later steps keep their durations";
  EXPECT_FALSE(guide.Update(At(start, 23.0)));
  EXPECT_TRUE(guide.Update(At(start, 24.0)));
  guide.Skip(At(start, 30.0));  // on the last step: nothing to skip to
  EXPECT_EQ(guide.current(), 2u);
}

TEST(TestGuide, SwitchingGuideIsComplete) {
  const std::vector<GuideStep> steps = SwitchingTestGuide();
  ASSERT_EQ(steps.size(), 10u);
  for (std::size_t i = 0; i + 1 < steps.size(); ++i) {
    EXPECT_GT(steps[i].seconds, 0.0) << steps[i].title;
    EXPECT_FALSE(steps[i].instruction.empty());
  }
  EXPECT_EQ(steps.back().seconds, 0.0);
}

}  // namespace
}  // namespace relume
