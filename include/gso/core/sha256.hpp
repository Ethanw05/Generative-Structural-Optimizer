#pragma once

// SHA-256 (FIPS 180-4). GSO identifies geometry and studies by content: a run manifest
// records the study's digest, checkpoints refuse to resume against a different study,
// and an evolved part records its parent's digest.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace gso {

using Digest = std::array<std::uint8_t, 32>;

// Hex encoding, lower case, 64 characters.
std::string toHex(const Digest& digest);

// Incremental hashing: feed any number of byte ranges, then call finish() once.
class Sha256 {
public:
    Sha256();

    void update(std::span<const std::byte> bytes);
    void update(std::string_view text);

    // Completes the hash. The object must not be updated afterwards.
    Digest finish();

private:
    void compressBlock(const std::uint8_t* block);

    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t bufferedBytes_ = 0;
    std::uint64_t totalBytes_ = 0;
    bool finished_ = false;
};

Digest sha256(std::string_view text);

}  // namespace gso
