# One helper defines every GSO module the same way: a static library named gso_<name>,
# aliased gso::<name>, with public headers under include/gso/<name>/ and sources under
# src/<name>/. Modules may only depend on modules below them in the layering documented
# in docs/port-plan.md; a module that reaches "up" fails to link.

function(gso_apply_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4 /WX /permissive- /utf-8 /external:W0 /Zc:__cplusplus /bigobj>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror>
        $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings -Xcompiler=/W4,/WX,/utf-8>)
    target_compile_definitions(${target} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES>)
endfunction()

# gso_module(<name> SOURCES <files...> [PUBLIC <targets...>] [PRIVATE <targets...>])
function(gso_module name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;PUBLIC;PRIVATE")
    set(target gso_${name})
    add_library(${target} STATIC ${arg_SOURCES})
    add_library(gso::${name} ALIAS ${target})
    target_include_directories(${target}
        PUBLIC "${PROJECT_SOURCE_DIR}/include"
        PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(${target} PUBLIC ${arg_PUBLIC} PRIVATE ${arg_PRIVATE})
    target_compile_features(${target} PUBLIC cxx_std_20)
    gso_apply_warnings(${target})
endfunction()
