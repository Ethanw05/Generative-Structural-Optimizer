#include "study.hpp"
#include "scenario_order.hpp"
#include "boundary_step.hpp"
#include "cad/stencil.hpp"
#ifdef GSO_NATIVE_CAD
#include "cad/native.hpp"
#include <comdef.h>
#endif
#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char **argv)
{
    if (argc == 5 && std::string_view(argv[1]) == "scenario-order")
    {
        try { gso::writeScenarioOrder(argv[2], argv[3], argv[4], std::cout); return 0; }
        catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    }
    if (argc == 6 && std::string_view(argv[1]) == "surface-motion")
    {
        try
        {
            gso::writeSurfaceMotion(argv[2], argv[3], std::stod(argv[4]), std::stod(argv[5]));
            return 0;
        }
        catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    }
    if (argc==5 && std::string_view(argv[1])=="boundary-step")
    {
        try { gso::writeBoundaryStep(argv[2],argv[3],std::stod(argv[4])); return 0; }
        catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    }
    if(argc==6 && std::string_view(argv[1])=="mls-stencil"){
        try {gso::writeMlsStencil(std::stod(argv[2]),std::stod(argv[3]),std::stod(argv[4]),argv[5]);return 0;}
        catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    }
#ifdef GSO_NATIVE_CAD
    if(argc==4 && std::string_view(argv[1])=="import") {
        try {gso::importNativeGeometry(argv[2],argv[3]);return 0;}
        catch(const _com_error& e){std::cerr<<"CAD automation error "<<std::hex<<e.Error()<<'\n';return 1;}
        catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    }
#endif
    if (argc != 3 || (std::string_view(argv[1]) != "analyze" && std::string_view(argv[1]) != "evolve"))
    {
        std::cerr << "Usage: gso import INPUT.SLDPRT|INPUT.SLDASM OUTPUT.obj\n       gso analyze|evolve STUDY.json\n";
        return 2;
    }
    try
    {
        return gso::runStudy(argv[1], argv[2], std::cout);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
