#include "gso/core/log.hpp"

#include <atomic>

namespace gso::log {
namespace {

// The sink is read on every message from any thread and replaced rarely, so it is held
// in an atomic shared_ptr: readers never block, and a sink being replaced stays alive
// until the last in-flight write that uses it finishes.
std::atomic<std::shared_ptr<LogSink>>& currentSink()
{
    static std::atomic<std::shared_ptr<LogSink>> sink;
    return sink;
}

}  // namespace

void setSink(std::shared_ptr<LogSink> sink)
{
    currentSink().store(std::move(sink));
}

void write(LogLevel level, std::string_view message)
{
    const std::shared_ptr<LogSink> sink = currentSink().load();
    if (sink) {
        sink->write(level, message);
    }
}

}  // namespace gso::log
