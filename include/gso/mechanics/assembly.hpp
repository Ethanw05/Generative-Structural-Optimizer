#pragma once
#include "gso/geometry/quadrature.hpp"
#include <Eigen/SparseCore>

namespace gso
{
using Sparse = Eigen::SparseMatrix<double>;
struct Material
{
    double modulusMPa;
    double poisson;
    double densityKgM3;
};
struct Restraint
{
    Point location;
    Point direction;
};
struct Force
{
    Point location;
    Point newtons;
};
struct Model
{
    ImplicitSolid solid;
    Material material;
    std::vector<Point> sites;
    double support;
    IntegrationSettings integration;
    std::vector<Restraint> restraints;
};
struct System
{
    Sparse stiffness, mass, constraints;
    std::vector<Point> sites;
    double massGrams = 0.0;
    double unresolvedVolume = 0.0;
    double lowerMassGrams = 0.0;
};
System assemble(const Model &model);
Eigen::VectorXd loadVector(const Model &model, const System &system, const std::vector<Force> &forces);
} // namespace gso
