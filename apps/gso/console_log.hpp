#pragma once

// Routes the library's log messages to the console through spdlog.

#include "gso/core/log.hpp"

namespace gso::app {

void installConsoleLog(bool verbose);

}  // namespace gso::app
