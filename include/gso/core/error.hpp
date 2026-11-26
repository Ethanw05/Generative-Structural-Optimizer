#pragma once

// Error conventions for all of GSO (docs/port-plan.md, section 7).
//
//   InputError       The user gave something unusable: a study field, a part, an argument.
//                    Carries the path of the offending item so the message can point at it.
//   EnvironmentError Something outside GSO failed: a file, the GPU driver, SOLIDWORKS.
//   InternalError    A broken invariant inside GSO. Always a bug. Raised by GSO_REQUIRE.
//   Cancelled        The run was stopped through its CancellationToken.
//
// Expected outcomes of a search (a rejected candidate, an unconverged trial step) are
// not errors and are returned as values by the code that produces them.

#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace gso {

class InputError : public std::runtime_error {
public:
    // `where` names the offending item, for example "cases[2].loads[0].on".
    InputError(std::string where, const std::string& message);

    const std::string& where() const noexcept { return where_; }
    // The message without the location prefix that what() includes.
    const std::string& message() const noexcept { return message_; }

private:
    std::string where_;
    std::string message_;
};

class EnvironmentError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class InternalError : public std::logic_error {
public:
    InternalError(const std::string& message, const std::source_location& location);

    const std::source_location& location() const noexcept { return location_; }

private:
    std::source_location location_;
};

class Cancelled : public std::runtime_error {
public:
    Cancelled() : std::runtime_error("The run was cancelled") {}
};

namespace detail {
[[noreturn]] void failRequirement(std::string_view condition,
                                  std::string_view message,
                                  const std::source_location& location);
}  // namespace detail

}  // namespace gso

// Checks an invariant in every build configuration. Numerical code must fail loudly and
// immediately rather than continue with a corrupt state, so this is never compiled out.
// A macro is used only to capture the text of the condition.
#define GSO_REQUIRE(condition, message)                                                     \
    do {                                                                                     \
        if (!(condition)) {                                                                  \
            ::gso::detail::failRequirement(#condition, (message),                           \
                                           std::source_location::current());                \
        }                                                                                    \
    } while (false)
