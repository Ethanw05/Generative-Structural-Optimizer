#include "dispatch.hpp"
#include "gso/optimization/evolution.hpp"
#include <algorithm>
#include <limits>
#include <optional>
#include <set>
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
    Json base = family.at("study");
    const auto &cfg = family.at("search");
    const auto &levels = family.at("levels");
    validateLevels(levels);
    Criteria criteria{cfg.at("relative_mass_change"), cfg.at("relative_response_change"), {}};
    struct Metric
    {
        std::string key, scenario, quantity;
        std::optional<std::size_t> region;
    };
    std::vector<Metric> metrics;
    for (const auto &load : base.at("load_cases"))
    {
        const std::string name = load.at("name");
        auto add = [&](const std::string &key, const std::string &quantity, double limit) {
            if (!std::isfinite(limit) || limit <= 0. || !criteria.responseLimits.emplace(key, limit).second)
            {
                throw std::invalid_argument("Invalid or duplicate response limit: " + key);
            }
            metrics.push_back({key, name, quantity, std::nullopt});
        };
        if (load.contains("max_work_Nmm"))
        {
            add(name, "complex_work_Nmm", load.at("max_work_Nmm"));
        }
        if (load.contains("limits"))
        {
            for (const auto &[quantity, limit] : load.at("limits").items())
            {
                const std::set<std::string> supported{"stress_p8_MPa",          "peak_von_mises_MPa",
                                                      "peak_probe_mm",          "peak_contact_force_N",
                                                      "peak_strain_energy_Nmm", "work_amplitude_Nmm",
                                                      "regional_peak_rms_MPa"};
                if (!supported.contains(quantity))
                {
                    throw std::invalid_argument("Unknown mechanical objective: " + quantity);
                }
                add(name + "/" + quantity, quantity, limit.get<double>());
            }
        }
        if (load.contains("stress_reference_mm") && load.contains("limits") && load.at("limits").contains("peak_von_mises_MPa"))
        {
            const double limit = load.at("limits").at("peak_von_mises_MPa");
            for (std::size_t i = 0; i < load.at("stress_reference_mm").size(); ++i)
            {
                add(name+"/local_peak/"+std::to_string(i), "tracked_stress_peaks_MPa", limit);
                metrics.back().region = i;
            }
        }
        if (metrics.empty() || metrics.back().scenario != name)
        {
            throw std::invalid_argument("Every scenario needs a response limit");
        }
    }
    SearchSettings settings{{}, cfg.at("generations"), cfg.at("parents"), cfg.at("evaluation_budget")};
    const std::string objective = cfg.value("objective", "minimum_mass");
    if (objective == "minimum_worst_response")
    {
        settings.objective = SearchObjective::minimumWorstResponse;
    }
    else if (objective != "minimum_mass")
    {
        throw std::invalid_argument("Unknown search objective");
    }
    settings.maxMassGrams = cfg.value("max_mass_g", std::numeric_limits<double>::infinity());
    for (const auto &a : cfg.at("axes"))
    {
        settings.axes.push_back({a.at("minimum"), a.at("maximum"), a.at("step")});
    }
    std::vector<double> initial;
    for (const auto &k : base.at("geometry").at("kernels"))
    {
        initial.push_back(k.at("amplitude"));
    }
    auto registerRegions = [&](Json &candidate, const Json &cases) {
        for (std::size_t i = 0; i < base.at("load_cases").size(); ++i)
        {
            auto &load = base.at("load_cases").at(i);
            if (!load.contains("limits") || !load.at("limits").contains("peak_von_mises_MPa") ||
                load.contains("stress_reference_mm")) { continue; }
            const std::string name = load.at("name");
            const auto &anchors = cases.at(name).at("stress_reference_mm");
            const double limit = load.at("limits").at("peak_von_mises_MPa");
            for (std::size_t region = 0; region < anchors.size(); ++region)
            {
                const auto key = name+"/local_peak/"+std::to_string(region);
                if (!criteria.responseLimits.emplace(key,limit).second)
                {
                    throw std::invalid_argument("Duplicate regional response identity");
                }
                metrics.push_back({key,name,"tracked_stress_peaks_MPa",region});
            }
            candidate.at("load_cases").at(i)["stress_reference_mm"] = anchors;
            load["stress_reference_mm"] = anchors;
        }
    };
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
            if (history.empty()) { registerRegions(candidate,run.at("load_cases")); }
            const double density = base.at("material").at("density_kg_m3");
            LevelResult level{l.at("spacing_mm"),
                              run.at("mass_g"),
                              density * 1e-6 * run.at("unresolved_boundary_volume_mm3").get<double>(),
                              run.at("mass_lower_g"),
                              {}};
            for (const auto &metric : metrics)
            {
                const auto &row = run.at("load_cases").at(metric.scenario);
                if (metric.quantity == "complex_work_Nmm")
                {
                    level.responses[metric.key] = {row.at("work_real_Nmm").get<double>(),
                                                   row.at("work_imag_Nmm").get<double>()};
                }
                else
                {
                    const auto &value = row.at(metric.quantity);
                    level.responses[metric.key] = metric.region ? value.at(*metric.region).get<double>() : value.get<double>();
                }
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
            Json responses = Json::object(), measured = Json::object();
            for (const auto &[name, v] : l.responses)
            {
                const auto metric = std::find_if(metrics.begin(), metrics.end(),
                                                 [&](const auto &m) { return m.key == name; });
                if (metric->quantity == "complex_work_Nmm")
                {
                    responses[name] = {{"real_Nmm", v.real()}, {"imag_Nmm", v.imag()}};
                }
                auto &quantity = measured[metric->scenario][metric->quantity];
                if (metric->region) { quantity[*metric->region] = std::abs(v); }
                else { quantity = std::abs(v); }
            }
            rows.push_back({{"spacing_mm", l.spacingMm},
                            {"mass_g", l.massGrams},
                            {"mass_lower_g", l.lowerMassGrams},
                            {"mass_upper_g", l.lowerMassGrams + l.unresolvedMassGrams},
                            {"unresolved_mass_g", l.unresolvedMassGrams},
                            {"work", responses},
                            {"responses", measured}});
        }
        trials.push_back({{"amplitudes", t.amplitudes},
                          {"parent", t.independentStart ? Json(nullptr) : Json(t.parent)},
                          {"independent_start", t.independentStart},
                          {"generation", t.generation},
                          {"decision", decisionName(t.decision)},
                          {"diagnostic", t.failure},
                          {"levels", rows}});
    }
    Json anchors = Json::object();
    for (const auto &load : base.at("load_cases"))
    {
        if (load.contains("stress_reference_mm")) { anchors[load.at("name").get<std::string>()] = load.at("stress_reference_mm"); }
    }
    const bool feasible = search.trials[search.best].decision == Decision::eligible;
    return {{"schema", "gso.evolution-result/1"},
            {"status", feasible ? "eligible_design" : "no_eligible_design"},
            {"selected", feasible ? Json(search.best) : Json(nullptr)},
            {"stress_reference_mm", std::move(anchors)},
            {"pareto_trials", search.paretoTrials},
            {"trials", trials}};
}
} // namespace gso
