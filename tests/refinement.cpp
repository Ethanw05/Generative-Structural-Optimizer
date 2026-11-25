#include "gso/optimization/refinement.hpp"
#include "test_support.hpp"

int main()
{
    using namespace gso;
    const Criteria criteria{.025, .03, {{"operating", 1.}}};
    std::vector<LevelResult> levels;
    auto add = [&](double h, double work) { levels.push_back({h, 1., .001, .9995, {{"operating", work}}}); };
    add(.5, .89);
    add(.25, .90);
    check(assess(levels, criteria) == Decision::refine, "Accepted a single refinement difference");
    add(.2, .91);
    check(assess(levels, criteria) == Decision::eligible, "Converged feasible case rejected");
    add(1. / 6., .98);
    check(assess(levels, criteria) == Decision::refine, "Missed a false plateau");
    levels.back().responses["operating"] = {0., .91};
    check(assess(levels, criteria) == Decision::refine, "Equal-amplitude phase shift missed");
    for (auto &l : levels)
    {
        l.responses["operating"] = 1.04;
    }
    check(assess(levels, criteria) == Decision::infeasible, "Over-limit response accepted");
    levels.back().unresolvedMassGrams = .10;
    check(assess(levels, criteria) == Decision::refine, "Unresolved boundary mass ignored");
    levels.back().spacingMm = .2;
    rejects([&] { assess(levels, criteria); }, "Duplicate analysis spacing accepted");
}
