#pragma once

// Reading study files. The format is JSON with an integer version; docs/study-format.md
// describes it. Reading is strict: unknown keys, missing keys, wrong types, references to
// undefined names and physically invalid values are all InputErrors naming the field,
// so a typo can never silently change a study.

#include "gso/study/study.hpp"

#include <filesystem>
#include <string_view>

namespace gso::study {

// The study-file version this build writes and the newest it reads.
inline constexpr int kCurrentFormatVersion = 1;

Study readStudy(const std::filesystem::path& path);

// Parses study text. Relative paths inside it are resolved against `baseDirectory`.
Study parseStudy(std::string_view text, const std::filesystem::path& baseDirectory);

}  // namespace gso::study
