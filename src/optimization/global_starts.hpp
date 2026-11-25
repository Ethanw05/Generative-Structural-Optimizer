#pragma once
#include "gso/optimization/evolution.hpp"

namespace gso
{
// Deterministic Halton starts spanning every bounded design coordinate.
class GlobalStarts
{
public:
    explicit GlobalStarts(std::size_t dimensions);
    std::vector<double> at(unsigned index, const std::vector<DesignAxis> &axes) const;

private:
    std::vector<unsigned> primes_;
};
} // namespace gso
