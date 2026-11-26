#include "cad/com_automation.hpp"

#include "gso/core/error.hpp"

#include <algorithm>
#include <format>
#include <utility>

namespace gso::cad::com {
namespace {

std::string narrow(const wchar_t* text)
{
    std::string result;
    for (const wchar_t* p = text; *p != L'\0'; ++p) {
        result.push_back(*p < 128 ? static_cast<char>(*p) : '?');
    }
    return result;
}

std::pair<LONG, LONG> arrayBounds(const _variant_t& value)
{
    if ((value.vt & VT_ARRAY) == 0 || value.parray == nullptr || SafeArrayGetDim(value.parray) != 1) {
        throw EnvironmentError("SOLIDWORKS returned something other than a one-dimensional array");
    }
    LONG lower = 0;
    LONG upper = 0;
    check(SafeArrayGetLBound(value.parray, 1, &lower), "Read array lower bound");
    check(SafeArrayGetUBound(value.parray, 1, &upper), "Read array upper bound");
    return {lower, upper};
}

}  // namespace

Apartment::Apartment()
{
    check(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED), "Initialise COM");
}

Apartment::~Apartment()
{
    CoUninitialize();
}

void check(HRESULT result, const char* context)
{
    if (FAILED(result)) {
        throw EnvironmentError(std::format("{} failed (HRESULT 0x{:08x})", context,
                                           static_cast<unsigned long>(result)));
    }
}

_variant_t call(const Dispatch& object, const wchar_t* member, std::initializer_list<_variant_t> arguments)
{
    if (!object) {
        throw EnvironmentError(std::format("Cannot call {}: missing SOLIDWORKS object", narrow(member)));
    }
    const std::string memberName = narrow(member);
    auto* name = const_cast<LPOLESTR>(member);
    DISPID id = 0;
    check(object->GetIDsOfNames(IID_NULL, &name, 1, LOCALE_USER_DEFAULT, &id), memberName.c_str());

    // IDispatch takes arguments in reverse order.
    std::vector<_variant_t> reversed(arguments);
    std::reverse(reversed.begin(), reversed.end());
    static_assert(sizeof(_variant_t) == sizeof(VARIANT), "_variant_t must be layout-compatible with VARIANT");
    DISPPARAMS parameters{reversed.empty() ? nullptr : static_cast<VARIANT*>(reversed.data()), nullptr,
                          static_cast<UINT>(reversed.size()), 0};

    _variant_t result;
    EXCEPINFO exception{};
    UINT badArgument = 0;
    const HRESULT status = object->Invoke(id, IID_NULL, LOCALE_USER_DEFAULT, DISPATCH_METHOD | DISPATCH_PROPERTYGET,
                                          &parameters, &result, &exception, &badArgument);
    SysFreeString(exception.bstrDescription);
    SysFreeString(exception.bstrHelpFile);
    SysFreeString(exception.bstrSource);
    check(status, memberName.c_str());
    return result;
}

Dispatch asObject(const _variant_t& value)
{
    Dispatch object;
    if (value.vt == VT_DISPATCH) {
        object = value.pdispVal;
    } else if (value.vt == VT_UNKNOWN && value.punkVal != nullptr) {
        check(value.punkVal->QueryInterface(IID_PPV_ARGS(&object)), "Convert SOLIDWORKS object");
    }
    return object;
}

Dispatch asInterface(const Dispatch& object, const wchar_t* interfaceId)
{
    IID iid{};
    check(IIDFromString(interfaceId, &iid), "Parse interface identifier");
    Dispatch result;
    check(object->QueryInterface(iid, reinterpret_cast<void**>(result.GetAddressOf())), "Query document interface");
    return result;
}

std::vector<Dispatch> asObjectArray(const _variant_t& value)
{
    if (value.vt == VT_EMPTY || value.vt == VT_NULL) {
        return {};
    }
    const auto [lower, upper] = arrayBounds(value);
    std::vector<Dispatch> objects;
    for (LONG i = lower; i <= upper; ++i) {
        if ((value.vt & VT_TYPEMASK) == VT_VARIANT) {
            _variant_t item;
            check(SafeArrayGetElement(value.parray, &i, &item), "Read array element");
            objects.push_back(asObject(item));
        } else if ((value.vt & VT_TYPEMASK) == VT_DISPATCH) {
            Dispatch item;
            check(SafeArrayGetElement(value.parray, &i, item.GetAddressOf()), "Read array element");
            objects.push_back(item);
        } else {
            throw EnvironmentError("SOLIDWORKS returned an object array of an unexpected type");
        }
    }
    return objects;
}

std::vector<double> asNumberArray(const _variant_t& value)
{
    const auto [lower, upper] = arrayBounds(value);
    const VARTYPE type = value.vt & VT_TYPEMASK;
    if (type != VT_R4 && type != VT_R8) {
        throw EnvironmentError("SOLIDWORKS returned a numeric array of an unexpected type");
    }
    const auto count = static_cast<std::size_t>(upper - lower + 1);
    std::vector<double> numbers(count);
    void* raw = nullptr;
    check(SafeArrayAccessData(value.parray, &raw), "Read array data");
    if (type == VT_R4) {
        std::copy_n(static_cast<const float*>(raw), count, numbers.begin());
    } else {
        std::copy_n(static_cast<const double*>(raw), count, numbers.begin());
    }
    check(SafeArrayUnaccessData(value.parray), "Release array data");
    return numbers;
}

std::wstring asString(const _variant_t& value)
{
    return static_cast<const wchar_t*>(_bstr_t(value));
}

}  // namespace gso::cad::com
