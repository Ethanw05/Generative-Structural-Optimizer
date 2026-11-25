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
    SearchSettings portfolio{{{-1., 1., .5}}, 4, 2, 15, SearchObjective::minimumWorstResponse, 3.1};
    auto ranked = [&](bool impact) {
        Criteria limits{.025, .03, {{"lift", 3.}}};
        if (impact)
        {
            limits.responseLimits.emplace("impact", 3.);
        }
        return evolve({0.}, portfolio, limits, [&](const std::vector<double> &x) {
            std::vector<LevelResult> rows;
            for (double h : {.4, .2, .1})
            {
                LevelResult row{h, 2., 0., 2., {{"lift", 1.5 + x.front()}}};
                if (impact)
                {
                    row.responses.emplace("impact", 1.5 - x.front());
                }
                rows.push_back(std::move(row));
            }
            return rows;
        });
    };
    auto lift = ranked(false), both = ranked(true);
    check(lift.paretoTrials.size() == 1, "Dominated single-objective candidates leaked into the archive");
    check(both.paretoTrials.size() > 2, "Distinct non-dominated scenario tradeoffs were discarded");
    near(lift.trials[lift.best].amplitudes.front(), -1., 1e-12, "Single-scenario optimum");
    near(both.trials[both.best].amplitudes.front(), 0., 1e-12,
         "Worst impact must change the selected design");
    // The local steps around zero all lead uphill. Only independent starts can
    // reach the narrow second basin before the local step size collapses.
    SearchSettings trapped{{{-1., 1., .02}}, 12, 1, 52, SearchObjective::minimumWorstResponse, 3.};
    const auto escaped = evolve({0.}, trapped, {.02, .03, {{"load", 4.}}}, [](const auto &field) {
        const double x = field.front();
        const double response = std::min(2. + x*x, .5 + 100.*(x-.75)*(x-.75));
        std::vector<LevelResult> levels;
        for (double h : {.4, .2, .1}) { levels.push_back({h, 1., 0., 1., {{"load", response}}}); }
        return levels;
    });
    near(escaped.trials[escaped.best].amplitudes.front(), .75, 1e-12, "Missed a separate better basin");
    check(escaped.trials[escaped.best].independentStart, "Global start provenance was lost");
}
