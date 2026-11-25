#pragma once
#include <complex>
#include <map>
#include <string>
#include <vector>

namespace gso
{
struct LevelResult
{
    double spacingMm;
    double massGrams;
    double unresolvedMassGrams;
    double lowerMassGrams;
    std::map<std::string, std::complex<double>> responses;
};
struct Criteria
{
    double relativeMass;
    double relativeResponse;
    std::map<std::string, double> responseLimits;
};
enum class Decision
{
    refine,
    infeasible,
    eligible
};
// Each level represents an independent reanalysis of one unchanged physical design and loading.
Decision assess(const std::vector<LevelResult> &levels, const Criteria &criteria);
} // namespace gso
