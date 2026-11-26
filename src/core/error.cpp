#include "gso/core/error.hpp"

#include <format>

namespace gso {

InputError::InputError(std::string where, const std::string& message)
    : std::runtime_error(where.empty() ? message : std::format("{}: {}", where, message)),
      where_(std::move(where)),
      message_(message)
{
}

InternalError::InternalError(const std::string& message, const std::source_location& location)
    : std::logic_error(message), location_(location)
{
}

namespace detail {

void failRequirement(std::string_view condition,
                     std::string_view message,
                     const std::source_location& location)
{
    throw InternalError(std::format("Internal error: {} (requirement `{}` failed in {} at {}:{})",
                                    message, condition, location.function_name(),
                                    location.file_name(), location.line()),
                        location);
}

}  // namespace detail
}  // namespace gso
