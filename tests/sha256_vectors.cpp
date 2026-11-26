// SHA-256 against the example messages of FIPS 180-2 (Appendix B) and the NIST
// "million a" vector. Together they cover one block, two blocks and many blocks.

#include "gso/core/sha256.hpp"

#include <doctest/doctest.h>

#include <string>

TEST_CASE("SHA-256 matches the FIPS 180-2 example digests")
{
    CHECK(gso::toHex(gso::sha256("")) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(gso::toHex(gso::sha256("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(gso::toHex(gso::sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST_CASE("SHA-256 of one million 'a' characters, fed in uneven pieces")
{
    gso::Sha256 hasher;
    const std::string piece(997, 'a');  // a prime length, so pieces straddle block boundaries
    std::size_t fed = 0;
    while (fed + piece.size() <= 1'000'000) {
        hasher.update(piece);
        fed += piece.size();
    }
    hasher.update(std::string(1'000'000 - fed, 'a'));
    CHECK(gso::toHex(hasher.finish()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}
