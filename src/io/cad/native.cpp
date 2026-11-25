#include "native.hpp"
#include "automation.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <vector>
namespace gso {
using namespace cad;
namespace {
struct Apartment {Apartment(){checked(CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED),"Initialize COM");} ~Apartment(){CoUninitialize();}};
struct CloseOwned {
    Dispatch app;_variant_t title;bool owned=false;
    ~CloseOwned(){if(owned){try{call(app,L"CloseDoc",{title});}catch(...){}}}
};
std::wstring wide(const _variant_t& v){return static_cast<const wchar_t*>(_bstr_t(v));}
struct Body {std::string name;std::vector<double> xyz;std::array<double,13> transform{1,0,0,0,1,0,0,0,1,0,0,0,1};};
void writeObj(const std::filesystem::path& output,const std::vector<Body>& bodies){
    if(bodies.empty()){throw std::runtime_error("Document has no resolved solid geometry");}
    const auto temp=std::filesystem::path(output.wstring()+L".gso-tmp-"+std::to_wstring(GetCurrentProcessId()));
    try {
        std::ofstream f(temp);f.exceptions(std::ios::badbit|std::ios::failbit);f<<"# GSO CAD surface. Coordinates in millimetres.\n"<<std::setprecision(12);
        std::size_t offset=0;
        for(const auto& b:bodies){
            if(b.xyz.empty() || b.xyz.size()%9){throw std::runtime_error("Incomplete CAD surface on "+b.name);}
            f<<"o "<<b.name<<'\n';const auto& t=b.transform;std::map<std::array<double,3>,std::size_t> ids;
            std::vector<std::size_t> faces;faces.reserve(b.xyz.size()/3);
            for(std::size_t j=0;j<b.xyz.size();j+=3){
                std::array<double,3> p{};
                for(int a=0;a<3;++a){p[a]=1000.*(t[9+a]+t[12]*(b.xyz[j]*t[a]+b.xyz[j+1]*t[3+a]+b.xyz[j+2]*t[6+a]));if(!std::isfinite(p[a])){throw std::runtime_error("Nonfinite CAD point");}}
                auto [it,added]=ids.emplace(p,offset+ids.size()+1);if(added){f<<"v "<<p[0]<<' '<<p[1]<<' '<<p[2]<<'\n';}faces.push_back(it->second);
            }
            const double det=t[12]*(t[0]*(t[4]*t[8]-t[5]*t[7])-t[1]*(t[3]*t[8]-t[5]*t[6])+t[2]*(t[3]*t[7]-t[4]*t[6]));
            for(std::size_t j=0;j<faces.size();j+=3){f<<"f "<<faces[j]<<' '<<faces[j+(det<0?2:1)]<<' '<<faces[j+(det<0?1:2)]<<'\n';}
            offset+=ids.size();
        }
        f.close();if(!MoveFileExW(temp.c_str(),output.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)){throw std::runtime_error("Could not publish OBJ");}
    }catch(...){std::error_code ignored;std::filesystem::remove(temp,ignored);throw;}
}
}
void importNativeGeometry(const std::filesystem::path& source,const std::filesystem::path& out){
    namespace fs=std::filesystem;auto path=fs::canonical(source);auto ext=path.extension().wstring();
    std::transform(ext.begin(),ext.end(),ext.begin(),::towlower);
    const long kind=ext==L".sldprt"?1L:ext==L".sldasm"?2L:0L;
    if(!kind){throw std::invalid_argument("Expected an SLDPRT or SLDASM input");}
    auto targetExt=out.extension().wstring();std::transform(targetExt.begin(),targetExt.end(),targetExt.begin(),::towlower);
    if(targetExt!=L".obj"){throw std::invalid_argument("Expected an OBJ output path");}
    Apartment apartment;CLSID clsid;checked(CLSIDFromProgID(L"SldWorks.Application",&clsid),"Installed SOLIDWORKS translator");
    Dispatch app;checked(CoCreateInstance(clsid,nullptr,CLSCTX_LOCAL_SERVER,IID_PPV_ARGS(&app)),"Start SOLIDWORKS geometry reader");
    auto name=_variant_t(path.c_str());auto model=object(call(app,L"GetOpenDocumentByName",{name}));
    CloseOwned lease{app,{},false};
    if(!model){long errors=0,warnings=0;_variant_t e,w;e.vt=w.vt=VT_BYREF|VT_I4;e.plVal=&errors;w.plVal=&warnings;
        model=object(call(app,L"OpenDoc6",{name,_variant_t(kind),_variant_t(3L),_variant_t(L""),e,w}));
        if(!model){throw std::runtime_error("Cannot open native geometry. SOLIDWORKS error "+std::to_string(errors));}
        lease.title=call(model,L"GetTitle");lease.owned=true;
    }
    std::vector<Body> bodies;
    if(kind==1){auto part=interfaceOf(model,L"{83A33D32-27C5-11CE-BFD4-00400513BB57}");
        bodies.push_back({path.stem().string(),numbers(call(part,L"GetTessTriangles",{_variant_t(true)}))});}
    else{
        auto assembly=interfaceOf(model,L"{83A33D35-27C5-11CE-BFD4-00400513BB57}");
        if(lease.owned){call(assembly,L"ResolveAllLightWeightComponents",{_variant_t(false)});}
        for(const auto& c:objects(call(assembly,L"GetComponents",{_variant_t(false)}))){
            if(static_cast<bool>(call(c,L"IsSuppressed"))){continue;}
            auto child=object(call(c,L"GetModelDoc2"));const auto nameW=wide(call(c,L"Name2"));
            if(!child){throw std::runtime_error("Unresolved assembly component: "+fs::path(nameW).string());}
            if(static_cast<long>(call(child,L"GetType"))==2L){continue;}
            auto geometry=call(c,L"GetTessTriangles",{_variant_t(true)});
            std::vector<double> xyz;if((geometry.vt&VT_ARRAY)!=0){xyz=numbers(geometry);}
            if(xyz.empty()){
                // Obtain the component's resolved, assembly-context bodies (including assembly cuts).
                _variant_t info,outInfo;outInfo.vt=VT_VARIANT|VT_BYREF;outInfo.pvarVal=&info;
                for(const auto& body:objects(call(c,L"GetBodies3",{_variant_t(0L),outInfo}))){
                    for(const auto& face:objects(call(body,L"GetFaces"))){
                        const auto triangle=call(face,L"GetTessTriangles",{_variant_t(true)});
                        if((triangle.vt&VT_ARRAY)==0){continue;}
                        auto faceCoords=numbers(triangle);xyz.insert(xyz.end(),faceCoords.begin(),faceCoords.end());
                    }
                }
            }
            Body b{fs::path(nameW).string(),std::move(xyz)};
            const auto tr=numbers(call(object(call(c,L"Transform2")),L"ArrayData"));if(tr.size()!=16){throw std::runtime_error("Invalid component placement");}
            std::copy_n(tr.begin(),13,b.transform.begin());bodies.push_back(std::move(b));
        }
    }
    writeObj(out,bodies);
}
}
