#pragma once
#include "gso/optimization/refinement.hpp"
#include <functional>
#include <limits>

namespace gso
{
struct DesignAxis
{
    double minimum, maximum, initialStep;
};
enum class SearchObjective
{
    minimumMass,
    minimumWorstResponse
};
struct SearchSettings
{
    std::vector<DesignAxis> axes;
    unsigned generations, parents, evaluationBudget;
    SearchObjective objective = SearchObjective::minimumMass;
    double maxMassGrams = std::numeric_limits<double>::infinity();
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
    bool independentStart = false;
};
struct SearchResult
{
    std::vector<Trial> trials;
    std::size_t best;
    std::vector<std::size_t> paretoTrials;
};
using Reanalysis = std::function<std::vector<LevelResult>(const std::vector<double> &)>;
SearchResult evolve(const std::vector<double> &initial, const SearchSettings &settings,
                    const Criteria &criteria, const Reanalysis &reanalyze);
} // namespace gso
