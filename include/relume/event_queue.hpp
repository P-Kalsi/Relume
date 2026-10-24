#pragma once

// Thread-safe, bounded queue of camera events. Capture threads and system
// notifications post; the main loop polls. Events are rare, so a mutex is
// fine; when full, the oldest event is discarded.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>

namespace relume {

struct CameraEvent {
  enum class Kind : std::uint8_t {
    kPermissionChanged,  // the permission prompt was answered
    kDevicesChanged,     // a camera was connected or disconnected
    kSourceStarted,      // the first frame arrived
    kSourceFailed,       // the source could not start or stopped with an error
    kSourceEnded,        // a finite source delivered its last frame
    kWarning,            // something worth showing, e.g. missing colour metadata
  };
  Kind kind = Kind::kWarning;
  std::string device_id;
  std::string message;
};

class CameraEventQueue {
 public:
  explicit CameraEventQueue(std::size_t capacity = 256) : capacity_(capacity) {}
  CameraEventQueue(const CameraEventQueue&) = delete;
  CameraEventQueue& operator=(const CameraEventQueue&) = delete;

  void Post(CameraEvent event);
  std::optional<CameraEvent> Poll();
  std::uint64_t discarded() const;

 private:
  const std::size_t capacity_;
  mutable std::mutex mutex_;
  std::deque<CameraEvent> events_;
  std::uint64_t discarded_ = 0;
};

}  // namespace relume
