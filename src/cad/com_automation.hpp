#pragma once

// Late-bound COM automation (IDispatch) helpers for talking to SOLIDWORKS. SOLIDWORKS
// exposes its API through IDispatch, so members are called by name.

#include <Windows.h>
#include <comdef.h>
#include <wrl/client.h>

#include <initializer_list>
#include <string>
#include <vector>

namespace gso::cad::com {

using Dispatch = Microsoft::WRL::ComPtr<IDispatch>;

// Initialises COM on this thread for the lifetime of the object.
class Apartment {
public:
    Apartment();
    ~Apartment();
    Apartment(const Apartment&) = delete;
    Apartment& operator=(const Apartment&) = delete;
};

// Throws EnvironmentError describing `context` if `result` is a failure code.
void check(HRESULT result, const char* context);

// Calls a method or reads a property by name. Arguments are given in natural order.
_variant_t call(const Dispatch& object, const wchar_t* member, std::initializer_list<_variant_t> arguments = {});

// Converts a VARIANT holding an object to an IDispatch pointer (null if it holds none).
Dispatch asObject(const _variant_t& value);

// Queries a specific interface by its IID string, e.g. "{83A33D32-...}".
Dispatch asInterface(const Dispatch& object, const wchar_t* interfaceId);

std::vector<Dispatch> asObjectArray(const _variant_t& value);
std::vector<double> asNumberArray(const _variant_t& value);
std::wstring asString(const _variant_t& value);

}  // namespace gso::cad::com
