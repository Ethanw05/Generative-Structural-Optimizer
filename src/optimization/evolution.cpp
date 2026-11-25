#include "gso/optimization/evolution.hpp"
#include "global_starts.hpp"
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
std::vector<double> responseUpperBounds(const Trial &trial)
{
    const auto &levels = trial.levels;
    const auto first = levels.size() > 3 ? levels.size() - 3 : 0;
    std::vector<double> values;
    for (const auto &[name, value] : levels.back().responses)
    {
        double upper = std::abs(value);
        for (auto i = first; i < levels.size(); ++i)
        {
            upper = std::max(upper, std::abs(levels[i].responses.at(name)));
        }
        values.push_back(upper);
    }
    return values;
}
bool noWorse(const std::vector<double> &a, const std::vector<double> &b)
{
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), std::less_equal<double>{});
}
bool better(const Trial &a, const Trial &b, SearchObjective objective)
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
    if ((a.decision == Decision::infeasible || objective == SearchObjective::minimumWorstResponse) &&
        a.worstRatio != b.worstRatio)
    {
        return a.worstRatio < b.worstRatio;
    }
    return massUpper(a) < massUpper(b);
}
} // namespace
SearchResult evolve(const std::vector<double> &initial, const SearchSettings &s, const Criteria &criteria,
                    const Reanalysis &reanalyze)
{
    if (!(s.maxMassGrams > 0.) || initial.empty() || initial.size() != s.axes.size() || s.parents == 0 ||
        s.parents > 32 || s.generations == 0 || s.generations > 10000 || s.evaluationBudget < 2 ||
        s.evaluationBudget > 1000000)
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
    SearchResult result{{}, 0, {}};
    std::set<std::vector<double>> evaluated;
    const GlobalStarts starts(initial.size());
    auto test = [&](const std::vector<double> &field, std::size_t parent, unsigned generation, bool independent = false) {
        if (result.trials.size() >= s.evaluationBudget || !evaluated.insert(field).second)
        {
            return;
        }
        Trial trial{field, parent, generation, Decision::refine, {}, 0., {}, independent};
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
            const auto count = trial.levels.size();
            if (trial.decision != Decision::refine && massUpper(trial) > s.maxMassGrams)
            {
                trial.decision = Decision::infeasible;
            }
            if (trial.decision == Decision::infeasible && count > 0)
            {
                trial.worstRatio = massUpper(trial) / s.maxMassGrams;
            }
            for (std::size_t i = count > 3 ? count - 3 : 0; i < count; ++i)
            {
                const auto &l = trial.levels[i];
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
        // The global sequence continues even when the local trust step shrinks.
        test(starts.at(generation, s.axes), 0, generation, true);
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
        std::stable_sort(pool.begin(), pool.end(), [&](auto a, auto b) {
            return better(result.trials[a], result.trials[b], s.objective);
        });
        if (pool.front() == parents.front())
        {
            scale *= .5;
        }
        pool.resize(std::min(pool.size(), static_cast<std::size_t>(s.parents)));
        parents = std::move(pool);
    }
    result.best = parents.front();
    if (s.objective == SearchObjective::minimumWorstResponse)
    {
        std::vector<std::vector<double>> values(result.trials.size());
        for (std::size_t i = 0; i < result.trials.size(); ++i)
        {
            if (result.trials[i].decision != Decision::eligible) { continue; }
            values[i] = responseUpperBounds(result.trials[i]);
            auto &front = result.paretoTrials;
            if (std::any_of(front.begin(), front.end(), [&](auto j){ return noWorse(values[j], values[i]); }))
            {
                continue;
            }
            std::erase_if(front, [&](auto j){ return noWorse(values[i], values[j]); });
            front.push_back(i);
        }
    }
    return result;
}
} // namespace gso
