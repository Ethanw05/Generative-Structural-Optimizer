#include "scenario.hpp"
#include "gso/mechanics/equilibrium.hpp"
#include "gso/mechanics/transient.hpp"
#include "gso/mechanics/wrench.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <stdexcept>
namespace gso
{
using Json = nlohmann::json;
namespace
{
std::vector<Force> applied(const Model &model, const Json &load)
{
    std::vector<Force> forces;
    for (const auto &f : load.value("forces", Json::array()))
    {
        forces.push_back({point(f.at("position_mm")), point(f.at("force_N"))});
    }
    for (const auto &t : load.value("tractions", Json::array()))
    {
        const auto values = integrateTraction(patch(t), point(t.at("traction_MPa")), model.support / 3.);
        forces.insert(forces.end(), values.begin(), values.end());
    }
    for (const auto &w : load.value("wrenches", Json::array()))
    {
        std::vector<AttachmentPoint> selection;
        for (const auto &entry : w.at("attachment"))
        {
            selection.push_back({point(entry.at("position_mm")), entry.value("weight", 1.)});
        }
        const auto values = distribute(
            {point(w.at("origin_mm")), point(w.at("force_N")), point(w.at("moment_Nmm"))}, selection);
        forces.insert(forces.end(), values.begin(), values.end());
    }
    return forces;
}
Point axis(const Json &d)
{
    const Point n = point(d);
    if (std::abs(n.norm() - 1.) > 1e-8)
    {
        throw std::invalid_argument("Expected unit direction");
    }
    return n;
}
Json localPeaks(const StressRecovery &recovery, const PeakSummary &summary, const std::vector<double> &times = {})
{
    Json peaks = Json::array();
    for (const auto &peak : summary.maxima)
    {
        const auto &x = recovery.location(peak.sample);
        Json row{{"world_mm", {x.x(), x.y(), x.z()}}, {"peak_MPa", peak.peakMPa},
                 {"basin_volume_mm3", peak.basinVolumeMm3}};
        if (!times.empty()) { row["time_s"] = times.at(peak.sample); }
        peaks.push_back(std::move(row));
    }
    return peaks;
}
void trackedPeaks(Json &out, const Json &scenario, const StressRecovery &recovery,
                  const std::vector<double> &field, const PeakSummary &peaks)
{
    std::vector<Point> anchors;
    if (scenario.contains("stress_reference_mm"))
    {
        for (const auto &entry : scenario.at("stress_reference_mm")) { anchors.push_back(point(entry)); }
    }
    else if (scenario.contains("limits") && scenario.at("limits").contains("peak_von_mises_MPa"))
    {
        Json positions = Json::array();
        for (const auto &peak : peaks.maxima)
        {
            const auto &x = recovery.location(peak.sample);
            anchors.push_back(x);
            positions.push_back({x.x(),x.y(),x.z()});
        }
        out["stress_reference_mm"] = std::move(positions);
    }
    if (!anchors.empty()) { out["tracked_stress_peaks_MPa"] = recovery.regionPeaks(field,anchors); }
}
Json history(const Model &m, const System &s, const Json &j, const StressRecovery *recovery)
{
    const auto n = s.mass.rows();
    TransientCase c{j.at("end_s"), j.at("max_step_s"), Eigen::VectorXd::Zero(n), {}, {}};
    c.massDampingPerSecond = j.value("rayleigh_mass_per_s", 0.);
    c.stiffnessDampingSeconds = j.value("rayleigh_stiffness_s", 0.);
    const Point initial = point(j.value("velocity_mm_s", Json::array({0, 0, 0})));
    const Point omega = point(j.value("angular_velocity_rad_s", Json::array({0, 0, 0})));
    const Point pivot = point(j.value("rotation_origin_mm", Json::array({0, 0, 0})));
    for (std::size_t i = 0; i < s.sites.size(); ++i)
    {
        c.initialVelocity.segment<3>(3 * i) = initial + omega.cross(s.sites[i] - pivot);
    }
    const auto steady = applied(m, j);
    if (!steady.empty())
    {
        c.loads.push_back({loadVector(m, s, steady), {{0., 1.}}});
    }
    if (j.contains("gravity_mm_s2"))
    {
        Eigen::VectorXd g(n);
        const auto acceleration = point(j.at("gravity_mm_s2"));
        for (Eigen::Index i = 0; i < n; i += 3)
        {
            g.segment<3>(i) = acceleration;
        }
        c.loads.push_back({s.mass * g, {{0., 1.}}});
    }
    for (const auto &h : j.value("time_loads", Json::array()))
    {
        TimeLoad l{loadVector(m, s, applied(m, h)), {}};
        for (const auto &k : h.at("history"))
        {
            l.history.push_back({k.at(0), k.at(1)});
        }
        c.loads.push_back(std::move(l));
    }
    for (const auto &plane : j.value("contact_planes", Json::array()))
    {
        const Point normal = axis(plane.at("normal")), origin = point(plane.at("origin_mm"));
        const double stiffness = plane.at("foundation_N_per_mm3");
        if (!std::isfinite(stiffness) || stiffness <= 0.)
        {
            throw std::invalid_argument("Invalid foundation stiffness");
        }
        for (const auto &sample : plane.at("samples"))
        {
            const Point x = point(sample.at("position_mm"));
            const double area = sample.at("area_mm2");
            if (!std::isfinite(area) || area <= 0.)
            {
                throw std::invalid_argument("Invalid contact area");
            }
            const Eigen::VectorXd row = loadVector(m, s, {{x, normal}});
            c.contacts.push_back({row.sparseView(), (x - origin).dot(normal), area * stiffness});
        }
    }
    Eigen::VectorXd probe;
    if (j.contains("probe"))
    {
        const auto &p = j.at("probe");
        probe = loadVector(m, s, {{point(p.at("position_mm")), axis(p.at("direction"))}});
    }
    const double tol = j.value("relative_time_tolerance", .02);
    if (!std::isfinite(tol) || tol <= 0. || tol >= 1.)
    {
        throw std::invalid_argument("Invalid time convergence tolerance");
    }
    std::vector<double> previous;
    unsigned stable = 0;
    for (unsigned level = 0; level < 7; ++level)
    {
        double stress = 0., peakStress = 0.;
        std::vector<double> field, envelope, times;
        const auto observer = [&](double time, const Eigen::VectorXd &u) {
            if (!recovery) { return; }
            const auto measure = recovery->measure(u, 8., &field);
            stress = std::max(stress, measure.aggregateMPa);
            peakStress = std::max(peakStress, measure.peakMPa);
            if (envelope.empty()) { envelope.assign(field.size(), 0.); times.assign(field.size(), 0.); }
            for (std::size_t i = 0; i < field.size(); ++i)
            {
                if (field[i] > envelope[i]) { envelope[i] = field[i]; times[i] = time; }
            }
        };
        const auto r = integrateTimedHistory(s, c, probe, observer);
        std::vector<double> metrics{r.peakStrainEnergyNmm, r.peakContactForceN, r.peakProbeMm, stress, peakStress};
        // Compare each material point across time refinements. A delayed smaller
        // impact peak can converge more slowly than the overall response.
        metrics.insert(metrics.end(), envelope.begin(), envelope.end());
        bool close = !previous.empty();
        if (close)
        {
            for (std::size_t i = 0; i < metrics.size(); ++i)
            {
                close =
                    close && std::abs(metrics[i] - previous[i]) <= tol * std::max(metrics[i], previous[i]);
            }
        }
        stable = close ? stable + 1 : 0;
        if (stable >= 2)
        {
            if (!c.contacts.empty() && j.value("require_impact", true) && r.peakContactForceN <= 0.)
            {
                throw std::runtime_error("Impact scenario ended without contact");
            }
            Json out = {{"peak_strain_energy_Nmm", metrics[0]},
                        {"peak_contact_force_N", metrics[1]},
                        {"time_steps", r.steps},
                        {"max_step_s", c.maxStepSeconds},
                        {"relative_residual", r.maxResidual}};
            if (probe.size())
            {
                out["peak_probe_mm"] = metrics[2];
            }
            if (recovery)
            {
                const auto peaks = recovery->peaks(envelope);
                trackedPeaks(out,j,*recovery,envelope,peaks);
                out["local_stress_peaks"] = localPeaks(*recovery, peaks, times);
                out["regional_peak_rms_MPa"] = peaks.regionalRmsMPa;
                out["stress_p8_MPa"] = stress;
                out["peak_von_mises_MPa"] = peakStress;
            }
            return out;
        }
        previous = std::move(metrics);
        c.maxStepSeconds *= .5;
    }
    throw std::runtime_error("Transient response did not converge on time refinement");
}
} // namespace
Json evaluateCase(const Model &m, const System &s, const Json &j, const StressRecovery *recovery)
{
    const std::string kind = j.at("kind");
    if (kind == "transient")
    {
        return history(m, s, j, recovery);
    }
    Excitation drive;
    if (kind == "harmonic")
    {
        drive = {j.at("frequency_Hz"), j.at("rayleigh_mass_per_s"), j.at("rayleigh_stiffness_s")};
        if (drive.hertz <= 0.)
        {
            throw std::invalid_argument("Harmonic frequency must be positive");
        }
    }
    else if (kind != "static")
    {
        throw std::invalid_argument("Unsupported load kind");
    }
    const auto r = solve(s, loadVector(m, s, applied(m, j)), drive);
    Json out = {{"work_real_Nmm", r.workNmm.real()},
                {"work_imag_Nmm", r.workNmm.imag()},
                {"work_amplitude_Nmm", std::abs(r.workNmm)},
                {"relative_residual", r.relativeResidual}};
    if (recovery)
    {
        std::vector<double> field;
        const auto stress = recovery->measureHarmonic(r.coefficients, 8., &field);
        const auto peaks = recovery->peaks(field);
        trackedPeaks(out,j,*recovery,field,peaks);
        out["local_stress_peaks"] = localPeaks(*recovery, peaks);
        out["regional_peak_rms_MPa"] = peaks.regionalRmsMPa;
        out["stress_p8_MPa"] = stress.aggregateMPa;
        out["peak_von_mises_MPa"] = stress.peakMPa;
    }
    return out;
}
} // namespace gso
