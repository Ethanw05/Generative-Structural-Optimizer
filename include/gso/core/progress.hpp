#pragma once

// Progress reporting and cancellation for long runs (docs/port-plan.md, section 7).
//
// A run receives a RunObserver to report to and a CancellationToken to poll. Long loops
// call `token.throwIfCancelled()` at every iteration boundary; the run then writes its
// checkpoint and returns with a Cancelled status.

#include <atomic>
#include <cstddef>
#include <memory>
#include <string_view>

namespace gso {

// A cancellation flag shared by every copy of the token. The host keeps one copy and
// calls requestCancel() from any thread (for example a UI button); the run polls another.
class CancellationToken {
public:
    CancellationToken();

    void requestCancel() noexcept;
    bool cancelRequested() const noexcept;

    // Throws gso::Cancelled if cancellation was requested.
    void throwIfCancelled() const;

private:
    std::shared_ptr<std::atomic<bool>> requested_;
};

struct ProgressUpdate {
    std::string_view stage;    // e.g. "topology"
    std::size_t done = 0;      // completed units of work
    std::size_t total = 0;     // 0 if the total is not known in advance
    std::string_view message;  // short human-readable detail
};

// Implemented by the host. Callbacks arrive on the run's worker thread and must return
// quickly: copy what is needed and hand it to the host's own thread.
class RunObserver {
public:
    virtual ~RunObserver() = default;
    virtual void onStage(std::string_view stage) = 0;
    virtual void onProgress(const ProgressUpdate& update) = 0;
};

// An observer that ignores everything, for runs with no host attached.
class SilentObserver final : public RunObserver {
public:
    void onStage(std::string_view) override {}
    void onProgress(const ProgressUpdate&) override {}
};

}  // namespace gso
