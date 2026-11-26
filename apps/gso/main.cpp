// gso: the command-line front end. It only parses arguments, calls the library and maps
// errors to exit codes; all behaviour lives in the library so a viewport can reuse it.
//
// Exit codes: 0 success, 1 invalid input, 2 environment failure (files, GPU, SOLIDWORKS),
// 3 internal error (a bug), 4 cancelled.

#include "console_log.hpp"

#include "gso/cad/part_file.hpp"
#include "gso/core/error.hpp"
#include "gso/core/version.hpp"
#include "gso/geometry/obj_file.hpp"
#include "gso/solvers/gpu_device.hpp"
#include "gso/study/study_file.hpp"

#include <CLI/CLI.hpp>

#include <cstdio>
#include <format>
#include <iostream>

namespace {

using namespace gso;

void printLine(const std::string& text)
{
    std::cout << text << '\n';
}

// gso info: build and hardware summary.
void runInfo()
{
    printLine(std::format("GSO {}", kVersion));
    printLine(std::format("CUDA {} (driver supports up to {})", solvers::cudaBuildVersion(),
                          solvers::cudaDriverVersion()));
    const solvers::GpuDevice gpu = solvers::selectGpu();
    printLine(std::format("GPU {}: {}, {:.1f} GiB, compute capability {}.{}, {} multiprocessors", gpu.index,
                          gpu.name, static_cast<double>(gpu.memoryBytes) / (1024.0 * 1024.0 * 1024.0),
                          gpu.computeCapabilityMajor, gpu.computeCapabilityMinor, gpu.multiprocessors));
}

// gso check STUDY: read and validate a study file without running it.
void runCheck(const std::filesystem::path& studyPath)
{
    const study::Study s = study::readStudy(studyPath);
    printLine(std::format("{} is valid (format version {}, sha256 {})", studyPath.filename().string(),
                          s.formatVersion, toHex(s.digest)));
    printLine(std::format("  part:       {}", s.part.string()));
    printLine(std::format("  selections: {}", s.selections.size()));
    printLine(std::format("  materials:  {}", s.materials.size()));
    printLine(std::format("  cases:      {}", s.cases.size()));
    printLine(std::format("  targets:    {}", s.targets.size()));
    if (!std::filesystem::exists(s.part)) {
        throw InputError("part.source", std::format("{} does not exist", s.part.string()));
    }
}

// gso import PART OUTPUT.obj: read a part in any supported format and write it as OBJ.
void runImport(const std::filesystem::path& part, const std::filesystem::path& output)
{
    const std::vector<geometry::NamedMesh> bodies = cad::readPart(part);
    for (const geometry::NamedMesh& body : bodies) {
        printLine(std::format("{}: {} triangles, volume {:.6g} mm^3", body.name, body.mesh.triangles.size(),
                              geometry::signedVolume(body.mesh)));
    }
    geometry::writeObj(output, bodies);
    printLine(std::format("wrote {}", output.string()));
}

int reportFailure(const char* kind, const std::exception& error, int code)
{
    std::cerr << kind << ": " << error.what() << '\n';
    return code;
}

}  // namespace

int main(int argc, char** argv)
{
    CLI::App app{"GSO generative structural optimizer"};
    app.require_subcommand(1);
    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Show debug messages");

    CLI::App* info = app.add_subcommand("info", "Show the version and the GPU that will be used");

    std::filesystem::path studyPath;
    CLI::App* check = app.add_subcommand("check", "Validate a study file");
    check->add_option("study", studyPath, "Study file (.json)")->required();

    std::filesystem::path partPath;
    std::filesystem::path outputPath;
    CLI::App* import = app.add_subcommand("import", "Convert a part (OBJ, SLDPRT, SLDASM) to OBJ");
    import->add_option("part", partPath, "Part file")->required();
    import->add_option("output", outputPath, "Output OBJ file")->required();

    CLI11_PARSE(app, argc, argv);
    app::installConsoleLog(verbose);

    try {
        if (*info) {
            runInfo();
        } else if (*check) {
            runCheck(studyPath);
        } else if (*import) {
            runImport(partPath, outputPath);
        }
        return 0;
    } catch (const InputError& error) {
        return reportFailure("input error", error, 1);
    } catch (const EnvironmentError& error) {
        return reportFailure("environment error", error, 2);
    } catch (const InternalError& error) {
        return reportFailure("internal error", error, 3);
    } catch (const Cancelled& error) {
        return reportFailure("cancelled", error, 4);
    } catch (const std::exception& error) {
        return reportFailure("internal error", error, 3);
    }
}
