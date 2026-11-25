#include "automation.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>
namespace gso::cad {
void checked(HRESULT hr,const char* context) {
    if(FAILED(hr)){std::ostringstream s;s<<context<<" (HRESULT 0x"<<std::hex<<static_cast<unsigned long>(hr)<<")";throw std::runtime_error(s.str());}
}
_variant_t call(const Dispatch& o,const wchar_t* member,std::initializer_list<_variant_t> args) {
    if(!o){throw std::runtime_error("Missing CAD automation object");}
    auto name=const_cast<LPOLESTR>(member);DISPID id=0;
    checked(o->GetIDsOfNames(IID_NULL,&name,1,LOCALE_USER_DEFAULT,&id),"Resolve CAD method");
    std::vector<_variant_t> reverse(args);std::reverse(reverse.begin(),reverse.end());
    static_assert(sizeof(_variant_t)==sizeof(VARIANT));
    DISPPARAMS params{reverse.empty()?nullptr:static_cast<VARIANT*>(reverse.data()),nullptr,static_cast<UINT>(reverse.size()),0};
    _variant_t result;EXCEPINFO exc{};UINT bad=0;
    const HRESULT hr=o->Invoke(id,IID_NULL,LOCALE_USER_DEFAULT,DISPATCH_METHOD|DISPATCH_PROPERTYGET,&params,&result,&exc,&bad);
    SysFreeString(exc.bstrDescription);SysFreeString(exc.bstrHelpFile);SysFreeString(exc.bstrSource);
    std::string label;for(const wchar_t* p=member;*p;++p){label.push_back(static_cast<char>(*p));}checked(hr,label.c_str());return result;
}
Dispatch object(const _variant_t& v) {
    Dispatch d;if(v.vt==VT_DISPATCH){d=v.pdispVal;}else if(v.vt==VT_UNKNOWN && v.punkVal){checked(v.punkVal->QueryInterface(IID_PPV_ARGS(&d)),"CAD object");}return d;
}
Dispatch interfaceOf(const Dispatch& d,const wchar_t* id) {
    IID iid;checked(IIDFromString(id,&iid),"CAD interface identifier");Dispatch out;
    checked(d->QueryInterface(iid,reinterpret_cast<void**>(out.GetAddressOf())),"CAD document interface");return out;
}
namespace {
std::pair<LONG,LONG> bounds(const _variant_t& v){
    if((v.vt&VT_ARRAY)==0 || !v.parray || SafeArrayGetDim(v.parray)!=1){throw std::runtime_error("Unexpected CAD array");}
    LONG lo,hi;checked(SafeArrayGetLBound(v.parray,1,&lo),"Array lower bound");checked(SafeArrayGetUBound(v.parray,1,&hi),"Array upper bound");return {lo,hi};
}
}
std::vector<Dispatch> objects(const _variant_t& v) {
    if(v.vt==VT_EMPTY || v.vt==VT_NULL){return {};}auto [lo,hi]=bounds(v);std::vector<Dispatch> out;
    for(LONG i=lo;i<=hi;++i){
        if((v.vt&VT_TYPEMASK)==VT_VARIANT){_variant_t item;checked(SafeArrayGetElement(v.parray,&i,&item),"Component");out.push_back(object(item));}
        else if((v.vt&VT_TYPEMASK)==VT_DISPATCH){Dispatch item;checked(SafeArrayGetElement(v.parray,&i,item.GetAddressOf()),"Component");out.push_back(item);}
        else{throw std::runtime_error("Unexpected CAD component array type");}
    }return out;
}
std::vector<double> numbers(const _variant_t& v) {
    auto [lo,hi]=bounds(v);const auto type=v.vt&VT_TYPEMASK;
    if(type!=VT_R4 && type!=VT_R8){throw std::runtime_error("Unsupported CAD numeric array");}
    const auto count=static_cast<std::size_t>(hi-lo+1);std::vector<double> result(count);
    void* raw=nullptr;checked(SafeArrayAccessData(v.parray,&raw),"Read CAD coordinates");
    if(type==VT_R4){std::copy_n(static_cast<const float*>(raw),count,result.begin());}
    else{std::copy_n(static_cast<const double*>(raw),count,result.begin());}
    checked(SafeArrayUnaccessData(v.parray),"Release CAD coordinates");return result;
}
}
