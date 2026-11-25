#include "gso/optimization/evolution.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace gso
{
namespace
{
double massUpper(const Trial &trial)
{
    const auto &last = trial.levels.back();
    return last.lowerMassGrams + last.unresolvedMassGrams;
}
bool better(const Trial &a, const Trial &b)
{
    if (a.decision != b.decision)
    {
        return a.decision == Decision::eligible ||
               (a.decision == Decision::infeasible && b.decision == Decision::refine);
    }
    if (a.decision == Decision::refine)
    {
        return false;
    }
    if (a.decision == Decision::infeasible && a.worstRatio != b.worstRatio)
    {
        return a.worstRatio < b.worstRatio;
    }
    return massUpper(a) < massUpper(b);
}
} // namespace
SearchResult evolve(const std::vector<double> &initial, const SearchSettings &s, const Criteria &criteria,
                    const Reanalysis &reanalyze)
{
    if (initial.empty() || initial.size() != s.axes.size() || s.parents == 0 || s.parents > 32 ||
        s.generations == 0 || s.generations > 10000 || s.evaluationBudget < 2 || s.evaluationBudget > 1000000)
    {
        throw std::invalid_argument("Invalid evolutionary search dimensions or budget");
    }
    for (std::size_t i = 0; i < initial.size(); ++i)
    {
        const auto &axis = s.axes[i];
        if (!std::isfinite(initial[i]) || !std::isfinite(axis.minimum) || !std::isfinite(axis.maximum) ||
            axis.minimum >= axis.maximum || initial[i] < axis.minimum || initial[i] > axis.maximum ||
            !std::isfinite(axis.initialStep) || axis.initialStep <= 0.)
        {
            throw std::invalid_argument("Invalid field amplitude range");
        }
    }
    SearchResult result{{}, 0};
    std::set<std::vector<double>> evaluated;
    auto test = [&](const std::vector<double> &field, std::size_t parent, unsigned generation) {
        if (result.trials.size() >= s.evaluationBudget || !evaluated.insert(field).second)
        {
            return;
        }
        Trial trial{field, parent, generation, Decision::refine, {}, 0., {}};
        try
        {
            trial.levels = reanalyze(field);
        }
        catch (const std::runtime_error &error)
        {
            trial.failure = error.what();
        }
        if (trial.failure.empty())
        {
            trial.decision = assess(trial.levels, criteria);
            for (const auto &l : trial.levels)
            {
                for (const auto &[name, work] : l.responses)
                {
                    trial.worstRatio =
                        std::max(trial.worstRatio, std::abs(work) / criteria.responseLimits.at(name));
                }
            }
        }
        result.trials.push_back(std::move(trial));
    };
    test(initial, 0, 0);
    std::vector<std::size_t> parents{0};
    double scale = 1.;
    for (unsigned generation = 1; generation <= s.generations && result.trials.size() < s.evaluationBudget;
         ++generation)
    {
        std::vector<std::size_t> pool = parents;
        const auto first = result.trials.size();
        for (auto parent : parents)
        {
            const auto base = result.trials[parent].amplitudes;
            // Positive and negative mutations allow both deposition and removal at any active control.
            for (std::size_t i = 0; i < base.size() && result.trials.size() < s.evaluationBudget; ++i)
            {
                for (double direction : {-1., 1.})
                {
                    auto child = base;
                    child[i] = std::clamp(base[i] + direction * scale * s.axes[i].initialStep,
                                          s.axes[i].minimum, s.axes[i].maximum);
                    test(child, parent, generation);
                }
            }
            if (parents.size() > 1 && parent != parents.front())
            {
                auto child = base;
                const auto &other = result.trials[parents.front()].amplitudes;
                for (std::size_t i = generation % 2; i < child.size(); i += 2)
                {
                    child[i] = other[i];
                }
                test(child, parent, generation);
            }
        }
        for (auto i = first; i < result.trials.size(); ++i)
        {
            if (result.trials[i].decision != Decision::refine)
            {
                pool.push_back(i);
            }
        }
        std::stable_sort(pool.begin(), pool.end(),
                         [&](auto a, auto b) { return better(result.trials[a], result.trials[b]); });
        if (pool.front() == parents.front())
        {
            scale *= .5;
        }
        pool.resize(std::min(pool.size(), static_cast<std::size_t>(s.parents)));
        parents = std::move(pool);
    }
    result.best = parents.front();
    return result;
}
} // namespace gso
