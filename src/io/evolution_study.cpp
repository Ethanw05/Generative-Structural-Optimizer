#include "dispatch.hpp"
#include "gso/optimization/evolution.hpp"
#include <limits>
#include <stdexcept>

namespace gso
{
namespace
{
using Json = nlohmann::json;
std::string decisionName(Decision d)
{
    switch (d)
    {
    case Decision::eligible:
        return "eligible";
    case Decision::infeasible:
        return "infeasible";
    case Decision::refine:
        return "refinement_required";
    }
    throw std::logic_error("Unknown decision");
}
void validateLevels(const Json &levels)
{
    if (!levels.is_array() || levels.size() < 3)
    {
        throw std::invalid_argument("At least three analysis levels are required");
    }
    unsigned long long previousCount = 0;
    double h = std::numeric_limits<double>::infinity(), radius = h;
    std::array<unsigned, 3> divisions{};
    unsigned depth = 0;
    for (const auto &l : levels)
    {
        const auto count = l.at("sites_mm").size();
        const double spacing = l.at("spacing_mm"), support = l.at("support_mm");
        const auto next = l.at("integration").at("divisions").get<std::array<unsigned, 3>>();
        const unsigned nextDepth = l.at("integration").at("boundary_depth");
        if (count <= previousCount || !std::isfinite(spacing) || spacing <= 0 || spacing >= h ||
            !std::isfinite(support) || support <= 0 || support >= radius || nextDepth < depth ||
            (next == divisions && nextDepth == depth))
        {
            throw std::invalid_argument("Cloud and integration must both refine");
        }
        for (int a = 0; a < 3; ++a)
        {
            if (next[a] < divisions[a])
            {
                throw std::invalid_argument("Integration coarsened");
            }
        }
        h = spacing;
        radius = support;
        previousCount = count;
        divisions = next;
        depth = nextDepth;
    }
}
} // namespace
Json runEvolution(const Json &family)
{
    if (family.at("schema") != "gso.evolution/1")
    {
        throw std::invalid_argument("Unsupported evolution schema");
    }
    const auto &base = family.at("study");
    const auto &cfg = family.at("search");
    const auto &levels = family.at("levels");
    validateLevels(levels);
    Criteria criteria{cfg.at("relative_mass_change"), cfg.at("relative_response_change"), {}};
    for (const auto &load : base.at("load_cases"))
    {
        if (!criteria.responseLimits
                 .emplace(load.at("name").get<std::string>(), load.at("max_work_Nmm").get<double>())
                 .second)
        {
            throw std::invalid_argument("Duplicate response limit");
        }
    }
    SearchSettings settings{{}, cfg.at("generations"), cfg.at("parents"), cfg.at("evaluation_budget")};
    for (const auto &a : cfg.at("axes"))
    {
        settings.axes.push_back({a.at("minimum"), a.at("maximum"), a.at("step")});
    }
    std::vector<double> initial;
    for (const auto &k : base.at("geometry").at("kernels"))
    {
        initial.push_back(k.at("amplitude"));
    }
    auto reanalyze = [&](const std::vector<double> &field) {
        Json candidate = base;
        for (std::size_t i = 0; i < field.size(); ++i)
        {
            candidate["geometry"]["kernels"][i]["amplitude"] = field[i];
        }
        std::vector<LevelResult> history;
        for (const auto &l : levels)
        {
            for (const auto key : {"sites_mm", "support_mm", "integration"})
            {
                candidate[key] = l.at(key);
            }
            const auto run = analyze(candidate);
            const double density = base.at("material").at("density_kg_m3");
            LevelResult level{l.at("spacing_mm"),
                              run.at("mass_g"),
                              density * 1e-6 * run.at("unresolved_boundary_volume_mm3").get<double>(),
                              run.at("mass_lower_g"),
                              {}};
            for (const auto &[name, value] : run.at("load_cases").items())
            {
                level.responses[name] = {value.at("work_real_Nmm").get<double>(),
                                         value.at("work_imag_Nmm").get<double>()};
            }
            history.push_back(std::move(level));
        }
        return history;
    };
    auto search = evolve(initial, settings, criteria, reanalyze);
    Json trials = Json::array();
    for (const auto &t : search.trials)
    {
        Json rows = Json::array();
        for (const auto &l : t.levels)
        {
            Json responses = Json::object();
            for (const auto &[name, v] : l.responses)
            {
                responses[name] = {{"real_Nmm", v.real()}, {"imag_Nmm", v.imag()}};
            }
            rows.push_back({{"spacing_mm", l.spacingMm},
                            {"mass_g", l.massGrams},
                            {"mass_lower_g", l.lowerMassGrams},
                            {"mass_upper_g", l.lowerMassGrams + l.unresolvedMassGrams},
                            {"unresolved_mass_g", l.unresolvedMassGrams},
                            {"work", responses}});
        }
        trials.push_back({{"amplitudes", t.amplitudes},
                          {"parent", t.parent},
                          {"generation", t.generation},
                          {"decision", decisionName(t.decision)},
                          {"diagnostic", t.failure},
                          {"levels", rows}});
    }
    const bool feasible = search.trials[search.best].decision == Decision::eligible;
    return {{"schema", "gso.evolution-result/1"},
            {"status", feasible ? "eligible_design" : "no_eligible_design"},
            {"selected", feasible ? Json(search.best) : Json(nullptr)},
            {"trials", trials}};
}
} // namespace gso
