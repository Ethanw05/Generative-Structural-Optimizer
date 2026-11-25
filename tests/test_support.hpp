#pragma once
#include <cmath>
#include <stdexcept>
#include <string>
inline void check(bool pass, const std::string &message)
{
    if (!pass)
    {
        throw std::runtime_error(message);
    }
}
inline void near(double a, double b, double tol, const std::string &message)
{
    check(std::isfinite(a) && std::abs(a - b) <= tol, message);
}
template <class F> void rejects(F action, const std::string &message)
{
    try
    {
        action();
    }
    catch (const std::exception &)
    {
        return;
    }
    throw std::runtime_error(message);
}
