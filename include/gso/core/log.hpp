#pragma once

// The library never prints. It writes through this interface, and the host application
// (the CLI today, a viewport later) decides where messages go by installing a sink.
// Without a sink, messages are discarded.

#include <format>
#include <memory>
#include <string_view>

namespace gso {

enum class LogLevel { Trace, Debug, Info, Warning, Error };

class LogSink {
public:
    virtual ~LogSink() = default;
    // Called from any thread. Implementations must be thread-safe.
    virtual void write(LogLevel level, std::string_view message) = 0;
};

namespace log {

// Installs the process-wide sink. Passing nullptr discards all messages.
void setSink(std::shared_ptr<LogSink> sink);

void write(LogLevel level, std::string_view message);

template <typename... Args>
void info(std::format_string<Args...> format, Args&&... args)
{
    write(LogLevel::Info, std::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
void debug(std::format_string<Args...> format, Args&&... args)
{
    write(LogLevel::Debug, std::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
void warning(std::format_string<Args...> format, Args&&... args)
{
    write(LogLevel::Warning, std::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
void error(std::format_string<Args...> format, Args&&... args)
{
    write(LogLevel::Error, std::format(format, std::forward<Args>(args)...));
}

}  // namespace log
}  // namespace gso
