#include "gso/optimization/evolution.hpp"
#include "test_support.hpp"

int main()
{
    using namespace gso;
    const Criteria criteria{.025, .03, {{"load", .69}}};
    const SearchSettings settings{{{-1., 1., .5}}, 5, 2, 12};
    // A scalar manufactured fitness isolates bidirectional selection from errors in the mechanics.
    auto reference = [](const std::vector<double> &field) {
        const double mass = 2. + field.front(), work = 1. / mass;
        std::vector<LevelResult> history;
        for (double h : {.4, .2, .1})
        {
            history.push_back({h, mass, 0., mass, {{"load", work}}});
        }
        return history;
    };
    for (double start : {-1., 1.})
    {
        const auto result = evolve({start}, settings, criteria, reference);
        const auto &winner = result.trials[result.best];
        check(winner.decision == Decision::eligible, "Failed to repair or reduce material");
        near(winner.amplitudes.front(), -.5, 1e-12, "Wrong feasible optimum on the sampled amplitudes");
        check(result.trials.size() <= settings.evaluationBudget, "Search exceeded the budget");
    }
    const auto unsupported = evolve({0.}, settings, criteria, [](const auto &) -> std::vector<LevelResult> {
        throw std::runtime_error("Unsupported solid");
    });
    check(unsupported.trials[unsupported.best].decision != Decision::eligible, "Rejected mechanics accepted");
}
