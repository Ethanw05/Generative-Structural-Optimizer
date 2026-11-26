#include "gso/core/sha256.hpp"

#include "gso/core/error.hpp"

#include <bit>
#include <cstring>

namespace gso {
namespace {

// FIPS 180-4, section 4.2.2: the first 32 bits of the fractional parts of the cube roots
// of the first 64 primes.
constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// FIPS 180-4, section 5.3.3: the first 32 bits of the fractional parts of the square roots
// of the first 8 primes.
constexpr std::array<std::uint32_t, 8> kInitialState = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

std::uint32_t readBigEndian32(const std::uint8_t* bytes)
{
    return (std::uint32_t{bytes[0]} << 24) | (std::uint32_t{bytes[1]} << 16) |
           (std::uint32_t{bytes[2]} << 8) | std::uint32_t{bytes[3]};
}

}  // namespace

Sha256::Sha256() : state_(kInitialState) {}

void Sha256::compressBlock(const std::uint8_t* block)
{
    // Message schedule (FIPS 180-4, section 6.2.2, step 1).
    std::array<std::uint32_t, 64> w{};
    for (std::size_t t = 0; t < 16; ++t) {
        w[t] = readBigEndian32(block + 4 * t);
    }
    for (std::size_t t = 16; t < 64; ++t) {
        const std::uint32_t s0 = std::rotr(w[t - 15], 7) ^ std::rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
        const std::uint32_t s1 = std::rotr(w[t - 2], 17) ^ std::rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
        w[t] = w[t - 16] + s0 + w[t - 7] + s1;
    }

    // Compression (steps 2-4).
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (std::size_t t = 0; t < 64; ++t) {
        const std::uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const std::uint32_t choose = (e & f) ^ (~e & g);
        const std::uint32_t temp1 = h + sum1 + choose + kRoundConstants[t] + w[t];
        const std::uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(std::span<const std::byte> bytes)
{
    GSO_REQUIRE(!finished_, "SHA-256 updated after finish()");
    const auto* data = reinterpret_cast<const std::uint8_t*>(bytes.data());
    std::size_t remaining = bytes.size();
    totalBytes_ += remaining;

    while (remaining > 0) {
        const std::size_t take = std::min(remaining, buffer_.size() - bufferedBytes_);
        std::memcpy(buffer_.data() + bufferedBytes_, data, take);
        bufferedBytes_ += take;
        data += take;
        remaining -= take;
        if (bufferedBytes_ == buffer_.size()) {
            compressBlock(buffer_.data());
            bufferedBytes_ = 0;
        }
    }
}

void Sha256::update(std::string_view text)
{
    update(std::as_bytes(std::span(text.data(), text.size())));
}

Digest Sha256::finish()
{
    GSO_REQUIRE(!finished_, "SHA-256 finished twice");
    // Padding (FIPS 180-4, section 5.1.1): a single 1 bit, zeros, then the message
    // length in bits as a 64-bit big-endian integer, filling the last block.
    const std::uint64_t lengthBits = totalBytes_ * 8;
    std::array<std::uint8_t, 72> padding{};
    padding[0] = 0x80;
    const std::size_t padBytes = (bufferedBytes_ < 56) ? 56 - bufferedBytes_ : 120 - bufferedBytes_;
    for (std::size_t i = 0; i < 8; ++i) {
        padding[padBytes + i] = static_cast<std::uint8_t>(lengthBits >> (56 - 8 * i));
    }
    const std::uint64_t messageBytes = totalBytes_;
    update(std::as_bytes(std::span(padding.data(), padBytes + 8)));
    totalBytes_ = messageBytes;
    GSO_REQUIRE(bufferedBytes_ == 0, "SHA-256 padding did not end on a block boundary");
    finished_ = true;

    Digest digest{};
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t byte = 0; byte < 4; ++byte) {
            digest[4 * i + byte] = static_cast<std::uint8_t>(state_[i] >> (24 - 8 * byte));
        }
    }
    return digest;
}

Digest sha256(std::string_view text)
{
    Sha256 hasher;
    hasher.update(text);
    return hasher.finish();
}

std::string toHex(const Digest& digest)
{
    constexpr std::string_view kHexDigits = "0123456789abcdef";
    std::string hex;
    hex.reserve(2 * digest.size());
    for (const std::uint8_t byte : digest) {
        hex.push_back(kHexDigits[byte >> 4]);
        hex.push_back(kHexDigits[byte & 0x0f]);
    }
    return hex;
}

}  // namespace gso
