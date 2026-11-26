#include "gso/core/progress.hpp"

#include "gso/core/error.hpp"

namespace gso {

CancellationToken::CancellationToken() : requested_(std::make_shared<std::atomic<bool>>(false)) {}

void CancellationToken::requestCancel() noexcept
{
    requested_->store(true, std::memory_order_relaxed);
}

bool CancellationToken::cancelRequested() const noexcept
{
    return requested_->load(std::memory_order_relaxed);
}

void CancellationToken::throwIfCancelled() const
{
    if (cancelRequested()) {
        throw Cancelled();
    }
}

}  // namespace gso
