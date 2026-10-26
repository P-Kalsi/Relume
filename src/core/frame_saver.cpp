#include "relume/frame_saver.hpp"

#include "relume/image_io.hpp"
#include "relume/pixel_convert.hpp"

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <system_error>

namespace relume {

namespace fs = std::filesystem;

namespace {

// A path relative to the save directory that cannot leave it.
bool IsContainedRelative(const std::string& path) {
  const fs::path p(path);
  if (path.empty() || p.is_absolute() || p.has_root_name()) {
    return false;
  }
  for (const fs::path& part : p) {
    if (part == "..") {
      return false;
    }
  }
  return true;
}

}  // namespace

std::string FrameSaver::Validate(const FrameSaverOptions& options) {
  if (options.directory.empty()) {
    return "no directory for saved frames";
  }
  if (options.count > kMaxFrames) {
    return "the number of frames to save must be 1.." + std::to_string(kMaxFrames);
  }
  if (!(options.interval_s >= 0.0) || !(options.delay_s >= 0.0)) {
    return "the save interval and delay must be 0 or more seconds";
  }
  return {};
}

FrameSaver::FrameSaver(FrameSaverOptions options) : options_(std::move(options)) {
  error_ = Validate(options_);
  if (error_.empty()) {
    std::error_code ec;
    fs::create_directories(options_.directory, ec);
    if (ec || !fs::is_directory(options_.directory)) {
      error_ = "cannot create " + options_.directory + ": " + ec.message();
    }
  }
  if (!error_.empty()) {
    broken_ = true;
    failed_ = options_.count;  // automatic mode: nothing will be accepted, so done() is true
    accepted_ = options_.count;
    return;
  }
  char stamp[32] = {};
  const std::time_t now = std::time(nullptr);
  if (const std::tm* local = std::localtime(&now)) {  // main thread only, once
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", local);
  }
  prefix_ = stamp;
  writer_ = std::thread([this] { WriterLoop(); });
}

FrameSaver::~FrameSaver() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  if (writer_.joinable()) {
    writer_.join();
  }
}

void FrameSaver::Offer(const VideoFrame& frame, Clock::time_point now) {
  if (manual() || accepted_ >= options_.count || frame.frame_id == last_frame_id_) {
    return;
  }
  if (first_offer_ == Clock::time_point{}) {
    first_offer_ = now;
  }
  const auto since = [&](Clock::time_point t) { return std::chrono::duration<double>(now - t).count(); };
  if (since(first_offer_) < options_.delay_s) {
    return;
  }
  if (accepted_ > 0 && since(last_accept_) < options_.interval_s) {
    return;
  }
  last_frame_id_ = frame.frame_id;
  char name[96];
  std::snprintf(name, sizeof(name), "%s_%02u_f%llu.ppm", prefix_.c_str(), sequence_ + 1,
                static_cast<unsigned long long>(frame.frame_id));
  Job job;
  job.kind = Job::Kind::kWrite;
  job.path = (fs::path(options_.directory) / name).string();
  {
    std::lock_guard lock(mutex_);
    if (buffers_out_ >= kBuffers) {
      ++skipped_busy_;
      return;
    }
    if (!free_.empty()) {
      job.frame = std::move(free_.back());
      free_.pop_back();
    }
    job.frame = frame;  // copy assignment reuses the buffer's capacity
    ++buffers_out_;
    Queue(std::move(job));
  }
  wake_.notify_one();
  last_accept_ = now;
  ++accepted_;
  ++sequence_;
}

std::uint64_t FrameSaver::Enqueue(const VideoFrame& frame, const std::string& relative_path) {
  if (!manual() || !IsContainedRelative(relative_path)) {
    return 0;
  }
  Job job;
  job.kind = Job::Kind::kWrite;
  job.path = (fs::path(options_.directory) / relative_path).string();
  std::uint64_t id = 0;
  {
    std::lock_guard lock(mutex_);
    if (broken_) {
      return 0;
    }
    if (buffers_out_ >= kBuffers) {
      ++skipped_busy_;
      return 0;
    }
    if (!free_.empty()) {
      job.frame = std::move(free_.back());
      free_.pop_back();
    }
    job.frame = frame;  // copy assignment reuses the buffer's capacity
    ++buffers_out_;
    id = Queue(std::move(job));
  }
  wake_.notify_one();
  return id;
}

std::uint64_t FrameSaver::EnqueueRemoveDir(const std::string& relative_dir) {
  if (!manual() || !IsContainedRelative(relative_dir)) {
    return 0;
  }
  Job job;
  job.kind = Job::Kind::kRemoveDir;
  job.path = (fs::path(options_.directory) / relative_dir).string();
  std::uint64_t id = 0;
  {
    std::lock_guard lock(mutex_);
    if (broken_) {
      return 0;
    }
    id = Queue(std::move(job));
  }
  wake_.notify_one();
  return id;
}

std::uint64_t FrameSaver::EnqueueCommitDir(const std::string& staging_dir, const std::string& final_dir,
                                           std::uint32_t expected_files) {
  if (!manual() || !IsContainedRelative(staging_dir) || !IsContainedRelative(final_dir)) {
    return 0;
  }
  Job job;
  job.kind = Job::Kind::kCommitDir;
  job.path = (fs::path(options_.directory) / staging_dir).string();
  job.target = (fs::path(options_.directory) / final_dir).string();
  job.expected = expected_files;
  std::uint64_t id = 0;
  {
    std::lock_guard lock(mutex_);
    if (broken_) {
      return 0;
    }
    id = Queue(std::move(job));
  }
  wake_.notify_one();
  return id;
}

std::uint64_t FrameSaver::Queue(Job job) {
  job.id = next_job_++;
  const std::uint64_t id = job.id;
  pending_.push_back(std::move(job));
  return id;
}

std::vector<SaveResult> FrameSaver::TakeResults() {
  std::lock_guard lock(mutex_);
  std::vector<SaveResult> out;
  out.swap(results_);
  return out;
}

SaveResult FrameSaver::Execute(Job& job, std::vector<std::uint8_t>& rgb) {
  SaveResult result;
  result.job = job.id;
  result.path = job.path;
  std::error_code ec;
  switch (job.kind) {
    case Job::Kind::kWrite: {
      const fs::path dest(job.path);
      const fs::path temp = fs::path(job.path + ".partial");
      if (!ToRgb8(job.frame, rgb)) {
        result.error = "frame " + std::to_string(job.frame.frame_id) + " has an invalid layout";
      } else if (fs::create_directories(dest.parent_path(), ec); ec) {
        result.error = "cannot create " + dest.parent_path().string() + ": " + ec.message();
      } else {
        RgbImage image;
        image.width = job.frame.width;
        image.height = job.frame.height;
        image.pixels.swap(rgb);
        const bool wrote = WritePpm(temp.string(), image);
        image.pixels.swap(rgb);
        if (!wrote) {
          result.error = "could not write " + temp.string();
        } else if (fs::rename(temp, dest, ec); ec) {
          result.error = "could not replace " + dest.string() + ": " + ec.message();
        } else {
          result.ok = true;
        }
      }
      if (!result.ok) {
        fs::remove(temp, ec);
        // A failed retake must not leave the previous shot looking current.
        if (!fs::is_directory(dest, ec)) {
          fs::remove(dest, ec);
        }
      }
      break;
    }
    case Job::Kind::kRemoveDir:
      fs::remove_all(job.path, ec);
      result.ok = !ec;
      result.error = ec ? "cannot remove " + job.path + ": " + ec.message() : std::string();
      break;
    case Job::Kind::kCommitDir: {
      result.path = job.target;
      std::uint32_t files = 0;
      if (fs::is_directory(job.path, ec)) {
        for (const fs::directory_entry& entry : fs::directory_iterator(job.path, ec)) {
          files += entry.is_regular_file() && entry.path().extension() == ".ppm" ? 1u : 0u;
        }
      }
      if (files != job.expected) {
        result.error = "only " + std::to_string(files) + " of " + std::to_string(job.expected) +
                       " frames were written; nothing was kept";
      } else if (fs::remove_all(job.target, ec); ec) {
        result.error = "cannot replace " + job.target + ": " + ec.message();
      } else if (fs::rename(job.path, job.target, ec); ec) {
        result.error = "cannot move " + job.path + " to " + job.target + ": " + ec.message();
      } else {
        result.ok = true;
      }
      if (!result.ok) {
        std::error_code ignored;
        fs::remove_all(job.path, ignored);
      }
      break;
    }
  }
  return result;
}

void FrameSaver::WriterLoop() {
  std::vector<std::uint8_t> rgb;
  for (;;) {
    Job job;
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, [&] { return stop_ || !pending_.empty(); });
      if (pending_.empty()) {
        return;  // stopping, and everything queued is done
      }
      job = std::move(pending_.front());
      pending_.pop_front();
    }
    const SaveResult result = Execute(job, rgb);
    std::lock_guard lock(mutex_);
    if (job.kind == Job::Kind::kWrite) {
      if (result.ok) {
        ++written_;
        if (!manual()) {
          paths_.push_back(result.path);
        }
      } else {
        ++failed_;
      }
      free_.push_back(std::move(job.frame));
      --buffers_out_;
    }
    if (!result.ok) {
      error_ = result.error;
    }
    if (manual()) {
      results_.push_back(result);
    }
  }
}

bool FrameSaver::done() const {
  std::lock_guard lock(mutex_);
  return !manual() && written_ + failed_ >= options_.count;
}

std::uint32_t FrameSaver::written() const {
  std::lock_guard lock(mutex_);
  return written_;
}

std::uint32_t FrameSaver::failed() const {
  std::lock_guard lock(mutex_);
  return failed_;
}

std::uint32_t FrameSaver::skipped_busy() const {
  std::lock_guard lock(mutex_);
  return skipped_busy_;
}

std::string FrameSaver::error() const {
  std::lock_guard lock(mutex_);
  return error_;
}

std::vector<std::string> FrameSaver::paths() const {
  std::lock_guard lock(mutex_);
  return paths_;
}

}  // namespace relume
