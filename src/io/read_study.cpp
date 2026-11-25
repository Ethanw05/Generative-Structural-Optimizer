#include "dispatch.hpp"
#include "gso/mechanics/boundary.hpp"
#include "gso/mechanics/equilibrium.hpp"
#include "study.hpp"
#include <fstream>
#include <json.hpp>
#include <set>
#include <stdexcept>

namespace gso
{
namespace
{
using Json = nlohmann::json;
Point point(const Json &a)
{
    if (!a.is_array() || a.size() != 3)
    {
        throw std::invalid_argument("Expected three Cartesian coordinates");
    }
    Point out(a.at(0).get<double>(), a.at(1).get<double>(), a.at(2).get<double>());
    if (!out.allFinite())
    {
        throw std::invalid_argument("Nonfinite Cartesian vector");
    }
    return out;
}
SurfacePatch patch(const Json &v)
{
    return {point(v.at("origin_mm")), point(v.at("edge_u_mm")), point(v.at("edge_v_mm"))};
}
Box box(const Json &b)
{
    return {point(b.at("min_mm")), point(b.at("max_mm"))};
}
std::vector<Box> boxes(const Json &data, const char *key)
{
    std::vector<Box> result;
    if (data.contains(key))
    {
        for (const auto &b : data.at(key))
        {
            result.push_back(box(b));
        }
    }
    return result;
}
ImplicitSolid geometry(const Json &data)
{
    std::vector<Kernel> field;
    for (const auto &k : data.at("kernels"))
    {
        field.push_back({point(k.at("center_mm")), k.at("radius_mm"), k.at("amplitude")});
    }
    return {box(data.at("domain")), data.at("background"), std::move(field), boxes(data, "preserves"),
            boxes(data, "exclusions")};
}
} // namespace
Json analyze(const Json &data)
{
    if (data.at("schema") != "gso.study/1")
    {
        throw std::invalid_argument("Unsupported study schema");
    }
    const auto &m = data.at("material");
    const auto &q = data.at("integration");
    Model model{
        geometry(data.at("geometry")),
        {m.at("young_MPa"), m.at("poisson"), m.at("density_kg_m3")},
        {},
        data.at("support_mm"),
        {q.at("divisions").get<std::array<unsigned, 3>>(), q.at("boundary_depth"), q.at("max_visits")},
        {}};
    for (const auto &p : data.at("sites_mm"))
    {
        model.sites.push_back(point(p));
    }
    if (data.contains("restraints"))
    {
        for (const auto &c : data.at("restraints"))
        {
            model.restraints.push_back({point(c.at("position_mm")), point(c.at("direction"))});
        }
    }
    if (data.contains("restrained_faces"))
    {
        for (const auto &c : data.at("restrained_faces"))
        {
            std::vector<Point> directions;
            for (const auto &d : c.at("directions"))
            {
                directions.push_back(point(d));
            }
            const auto trace = collocateRestraints(model.sites, patch(c), directions);
            model.restraints.insert(model.restraints.end(), trace.begin(), trace.end());
        }
    }
    const auto system = assemble(model);
    Json cases = Json::object();
    for (const auto &load : data.at("load_cases"))
    {
        const std::string name = load.at("name"), kind = load.at("kind");
        if (name.empty() || cases.contains(name))
        {
            throw std::invalid_argument("Duplicate or empty load name");
        }
        Excitation drive;
        if (kind == "harmonic")
        {
            drive = {load.at("frequency_Hz"), load.at("rayleigh_mass_per_s"),
                     load.at("rayleigh_stiffness_s")};
            if (drive.hertz <= 0.)
            {
                throw std::invalid_argument("Harmonic frequency must be positive");
            }
        }
        else if (kind != "static")
        {
            throw std::invalid_argument("Unsupported load kind");
        }
        std::vector<Force> forces;
        if (load.contains("forces"))
        {
            for (const auto &f : load.at("forces"))
            {
                forces.push_back({point(f.at("position_mm")), point(f.at("force_N"))});
            }
        }
        if (load.contains("tractions"))
        {
            for (const auto &t : load.at("tractions"))
            {
                const auto distributed =
                    integrateTraction(patch(t), point(t.at("traction_MPa")), model.support / 3.);
                forces.insert(forces.end(), distributed.begin(), distributed.end());
            }
        }
        const auto result = solve(system, loadVector(model, system, forces), drive);
        cases[name] = {{"work_real_Nmm", result.workNmm.real()},
                       {"work_imag_Nmm", result.workNmm.imag()},
                       {"work_amplitude_Nmm", std::abs(result.workNmm)},
                       {"relative_residual", result.relativeResidual}};
    }
    if (cases.empty())
    {
        throw std::invalid_argument("At least one load case is required");
    }
    return Json{
        {"schema", "gso.analysis/1"},
        {"mass_g", system.massGrams},
        {"mass_lower_g", system.lowerMassGrams},
        {"mass_upper_g", system.lowerMassGrams + system.unresolvedVolume * model.material.densityKgM3 * 1e-6},
        {"unresolved_boundary_volume_mm3", system.unresolvedVolume},
        {"load_cases", std::move(cases)}};
}
int runStudy(std::string_view command, const std::filesystem::path &path, std::ostream &output)
{
    std::ifstream file(path);
    if (!file)
    {
        throw std::runtime_error("Cannot open study file");
    }
    const Json data = Json::parse(file);
    const Json result = command == "evolve" ? runEvolution(data) : analyze(data);
    output << result.dump(2) << '\n';
    return result.value("status", std::string()) == "no_eligible_design" ? 3 : 0;
}
} // namespace gso
