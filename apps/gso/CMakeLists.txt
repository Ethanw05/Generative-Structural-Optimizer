add_executable(gso main.cpp console_log.cpp)
target_link_libraries(gso PRIVATE gso::cad gso::study gso::solvers CLI11::CLI11 spdlog::spdlog)
gso_apply_warnings(gso)
