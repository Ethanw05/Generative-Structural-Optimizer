#include "scenario_order.hpp"
#include "surface_motion.hpp"
#include "cad/stencil.hpp"
#ifdef GSO_NATIVE_CAD
#include "cad/native.hpp"
#include <comdef.h>
#endif
#include <exception>
#include <iostream>
#include <string>
#include <string_view>

// Native helper commands called by the Python workflows in tools/gso_workflows.
int main(int argc, char **argv)
{
    const std::string_view command = argc > 1 ? argv[1] : "";
    try
    {
        if (argc == 5 && command == "scenario-order")
        {
            gso::writeScenarioOrder(argv[2], argv[3], argv[4], std::cout);
            return 0;
        }
        if (argc == 6 && command == "surface-motion")
        {
            gso::writeSurfaceMotion(argv[2], argv[3], std::stod(argv[4]), std::stod(argv[5]));
            return 0;
        }
        if (argc == 6 && command == "mls-stencil")
        {
            gso::writeMlsStencil(std::stod(argv[2]), std::stod(argv[3]), std::stod(argv[4]), argv[5]);
            return 0;
        }
#ifdef GSO_NATIVE_CAD
        if (argc == 4 && command == "import")
        {
            gso::importNativeGeometry(argv[2], argv[3]);
            return 0;
        }
#endif
    }
#ifdef GSO_NATIVE_CAD
    catch (const _com_error &e)
    {
        std::cerr << "CAD automation error " << std::hex << e.Error() << '\n';
        return 1;
    }
#endif
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
    std::cerr << "Usage: gso import INPUT.SLDPRT|INPUT.SLDASM OUTPUT.obj   (Windows with SOLIDWORKS)\n"
                 "       gso mls-stencil PITCH_MM YOUNG_MPA POISSON OUTPUT.txt\n"
                 "       gso surface-motion INPUT.txt OUTPUT.txt MOVE_BOUND_MM AVAILABLE_VOLUME_MM3\n"
                 "       gso scenario-order COUNT SEED SWEEP\n";
    return 2;
}
