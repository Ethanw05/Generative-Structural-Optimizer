#pragma once
#include "gso/optimization/refinement.hpp"
#include <functional>

namespace gso
{
struct DesignAxis
{
    double minimum, maximum, initialStep;
};
struct SearchSettings
{
    std::vector<DesignAxis> axes;
    unsigned generations, parents, evaluationBudget;
};
struct Trial
{
    std::vector<double> amplitudes;
    std::size_t parent;
    unsigned generation;
    Decision decision = Decision::refine;
    std::vector<LevelResult> levels;
    double worstRatio = 0.0;
    std::string failure;
};
struct SearchResult
{
    std::vector<Trial> trials;
    std::size_t best;
};
using Reanalysis = std::function<std::vector<LevelResult>(const std::vector<double> &)>;
SearchResult evolve(const std::vector<double> &initial, const SearchSettings &settings,
                    const Criteria &criteria, const Reanalysis &reanalyze);
} // namespace gso
