#pragma once
#include <Windows.h>
#include <comdef.h>
#include <wrl/client.h>
#include <initializer_list>
#include <vector>
namespace gso::cad {
using Dispatch=Microsoft::WRL::ComPtr<IDispatch>;
void checked(HRESULT hr,const char* context);
_variant_t call(const Dispatch& object,const wchar_t* member,std::initializer_list<_variant_t> args={});
Dispatch object(const _variant_t& variant);
Dispatch interfaceOf(const Dispatch& source,const wchar_t* iid);
std::vector<Dispatch> objects(const _variant_t& values);
std::vector<double> numbers(const _variant_t& values);
}
