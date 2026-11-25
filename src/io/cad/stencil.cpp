#include "stencil.hpp"
#include "gso/mechanics/assembly.hpp"
#include <fstream>
#include <iomanip>

namespace gso {
// Translation-invariant interior cubature for batched compliance proposals. A 1.22-pitch support
// at the eight Gauss locations excludes all lattice sites outside this eight-site neighbourhood.
void writeMlsStencil(double h,double modulus,double poisson,const std::filesystem::path& output) {
    Model m{ImplicitSolid({Point::Zero(),Point::Constant(h)},1.),{modulus,poisson,1.},
            {},1.22*h,{{1,1,1},0,1},{}};
    for(int i=0;i<2;++i){for(int j=0;j<2;++j){for(int k=0;k<2;++k){m.sites.emplace_back(i*h,j*h,k*h);}}}
    for(int i : {0,1,2}){for(int a=0;a<3;++a){m.restraints.push_back({m.sites[i],Point::Unit(a)});}}
    const auto system=assemble(m);
    std::ofstream stream(output);stream.exceptions(std::ios::badbit|std::ios::failbit);
    stream<<std::setprecision(17)<<Eigen::MatrixXd(system.stiffness)<<'\n';
}
}
