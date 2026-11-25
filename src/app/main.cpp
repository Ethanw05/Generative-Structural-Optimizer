#include "study.hpp"
#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char **argv)
{
    if (argc != 3 || (std::string_view(argv[1]) != "analyze" && std::string_view(argv[1]) != "evolve"))
    {
        std::cerr << "Usage: gso analyze|evolve STUDY.json\n";
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
