#include "gso/mechanics/finite_strain.hpp"
#include "test_support.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

namespace
{
using Tensor = Eigen::Matrix3d;
void checkTangent(const gso::SolidLaw& law, const Tensor& f, const gso::PlasticState& prior)
{
    const auto result=gso::finiteStrain(law,f,prior,true);
    for (int i=0; i<3; ++i)
    {
        for (int j=0; j<3; ++j)
        {
            Tensor df=Tensor::Zero();
            df(i,j)=1.e-6;
            const Tensor derivative=(gso::finiteStrain(law,f+df,prior,false).firstPiola-
                                     gso::finiteStrain(law,f-df,prior,false).firstPiola)/(2.e-6);
            for (int a=0; a<3; ++a)
                for (int b=0; b<3; ++b)
                    check(std::abs(derivative(a,b)-result.tangent(3*a+b,3*i+j))<2.e-5,
                          "Constitutive tangent failed centered differences");
        }
    }
}
}

int main()
{
    const gso::SolidLaw elastic{80.,160.,0.,0.}, plastic{80.,160.,3.,12.};
    const gso::PlasticState virgin;
    const Tensor spin=Eigen::AngleAxisd(1.3,Eigen::Vector3d(2.,1.,-3.).normalized()).matrix();
    const auto rigid=gso::finiteStrain(elastic,spin,virgin,true);
    check(rigid.firstPiola.norm()<1.e-12,"Rigid rotation generated stress");
    checkTangent(elastic,Tensor::Identity(),virgin);
    Tensor stretch=Tensor::Identity();
    stretch(0,0)=std::exp(.12);
    stretch(1,1)=std::exp(-.12);
    const auto load=gso::finiteStrain(plastic,stretch,virgin,true);
    check(load.next.equivalentStrain>0.,"Plastic increment missing");
    check(std::abs(load.vonMisesMPa-3.-12.*load.next.equivalentStrain)<1.e-10,"Incorrect yield surface");
    check(std::abs(load.next.inversePlasticMetric.determinant()-1.)<1.e-12,"Plastic flow changed volume");
    check((gso::finiteStrain(plastic,spin*stretch,virgin,false).firstPiola-spin*load.firstPiola).norm()<1.e-10,
          "Constitutive response lost objectivity");
    checkTangent(plastic,spin*stretch,virgin);
    const Eigen::SelfAdjointEigenSolver<Tensor> axes(load.next.inversePlasticMetric);
    const Tensor unloaded=axes.eigenvectors()*axes.eigenvalues().array().rsqrt().matrix().asDiagonal()*axes.eigenvectors().transpose();
    const auto release=gso::finiteStrain(plastic,unloaded,load.next,false);
    check(release.firstPiola.norm()<1.e-10 && (unloaded-Tensor::Identity()).norm()>.01,
          "Unloading did not leave the analytical permanent deformation");
    check(release.next.equivalentStrain==load.next.equivalentStrain,"Elastic unloading changed plastic history");
    checkTangent(plastic,unloaded,load.next);
    Tensor reloaded=spin*stretch;
    reloaded.col(1)+=.12*reloaded.col(0);
    checkTangent(plastic,reloaded,load.next);
    Tensor inverted=Tensor::Identity();
    inverted(0,0)=-1.;
    bool rejected=false;
    try { gso::finiteStrain(elastic,inverted,virgin,false); }
    catch (const std::domain_error&) { rejected=true; }
    check(rejected,"An inverted element was accepted");
}
