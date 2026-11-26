#include "console_log.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace gso::app {
namespace {

spdlog::level::level_enum toSpdlog(LogLevel level)
{
    switch (level) {
    case LogLevel::Trace:
        return spdlog::level::trace;
    case LogLevel::Debug:
        return spdlog::level::debug;
    case LogLevel::Info:
        return spdlog::level::info;
    case LogLevel::Warning:
        return spdlog::level::warn;
    case LogLevel::Error:
        return spdlog::level::err;
    }
    return spdlog::level::err;
}

class SpdlogSink final : public LogSink {
public:
    explicit SpdlogSink(std::shared_ptr<spdlog::logger> logger) : logger_(std::move(logger)) {}

    void write(LogLevel level, std::string_view message) override { logger_->log(toSpdlog(level), message); }

private:
    std::shared_ptr<spdlog::logger> logger_;
};

}  // namespace

void installConsoleLog(bool verbose)
{
    auto logger = spdlog::stderr_color_mt("gso");
    logger->set_pattern("%^[%l]%$ %v");
    logger->set_level(verbose ? spdlog::level::debug : spdlog::level::info);
    log::setSink(std::make_shared<SpdlogSink>(std::move(logger)));
}

}  // namespace gso::app
