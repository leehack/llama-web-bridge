option(LLAMADART_LLAMA_CPP_PATCH_QUALIFICATION
    "Warn and skip non-applicable llama.cpp patches when qualifying another upstream" OFF)

set(LLAMADART_PATCH_MODE "")
if (LLAMADART_LLAMA_CPP_PATCH_QUALIFICATION)
    set(LLAMADART_PATCH_MODE --qualification)
endif()

function(llamadart_apply_llama_cpp_patches bridge_dir patches_dir)
    find_program(LLAMADART_PATCH_NODE node REQUIRED)
    set(tool "${bridge_dir}/scripts/build/llama_cpp_patches.mjs")
    set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp-patched")
    file(GLOB patch_files CONFIGURE_DEPENDS "${patches_dir}/*.patch")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${patches_dir}/series.json" ${patch_files} "${tool}")

    execute_process(
        COMMAND "${LLAMADART_PATCH_NODE}" "${tool}" ${LLAMADART_PATCH_MODE}
            configure "${LLAMA_CPP_DIR}" "${patches_dir}" "${output_dir}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE configuration
        ERROR_VARIABLE diagnostics
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT result STREQUAL "0")
        message(FATAL_ERROR "Could not apply llama.cpp patches:\n${diagnostics}")
    endif()
    if (diagnostics)
        message(WARNING "${diagnostics}")
    endif()

    string(JSON source_count LENGTH "${configuration}" sources)
    set(index 0)
    while (index LESS source_count)
        string(JSON relative GET "${configuration}" sources "${index}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LLAMA_CPP_DIR}/${relative}")
        math(EXPR index "${index} + 1")
    endwhile()

    string(JSON replacement_count LENGTH "${configuration}" replacements)
    set(index 0)
    while (index LESS replacement_count)
        string(JSON relative GET "${configuration}" replacements "${index}" relative)
        string(JSON target GET "${configuration}" replacements "${index}" target)
        llamadart_replace_llama_cpp_source("${target}" "${relative}" "${output_dir}")
        math(EXPR index "${index} + 1")
    endwhile()
endfunction()

function(llamadart_replace_llama_cpp_source target relative output_dir)
    set(original "${LLAMA_CPP_DIR}/${relative}")
    if (NOT TARGET "${target}")
        message(FATAL_ERROR "llama.cpp patches: ${target} does not exist for ${relative}")
    endif()
    get_target_property(source_dir "${target}" SOURCE_DIR)
    get_target_property(sources "${target}" SOURCES)
    set(matches 0)
    set(replacements "")
    foreach(source IN LISTS sources)
        if (NOT source MATCHES "^\\$<")
            get_filename_component(absolute "${source}" ABSOLUTE BASE_DIR "${source_dir}")
            if (absolute STREQUAL original)
                set(source "${output_dir}/${relative}")
                math(EXPR matches "${matches} + 1")
            endif()
        endif()
        list(APPEND replacements "${source}")
    endforeach()
    if (NOT matches EQUAL 1)
        message(FATAL_ERROR "llama.cpp patches: ${target} must compile ${relative} exactly once (found ${matches})")
    endif()
    set_property(TARGET "${target}" PROPERTY SOURCES "${replacements}")
    # A unity build would compile the copy inside another file's translation
    # unit and drop the options that apply to it alone.
    set_source_files_properties("${output_dir}/${relative}"
        DIRECTORY "${source_dir}" PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    # The relocated source no longer finds its private headers next to itself.
    get_filename_component(original_dir "${original}" DIRECTORY)
    target_include_directories("${target}" BEFORE PRIVATE "${original_dir}")
    message(STATUS "llama.cpp patches: ${target} compiles patched ${relative}")
endfunction()

function(llamadart_verify_llama_cpp_patches target)
    add_custom_target(llama_cpp_patch_markers ALL
        COMMAND "${LLAMADART_PATCH_NODE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build/llama_cpp_patches.mjs"
            ${LLAMADART_PATCH_MODE} verify "${LLAMA_CPP_DIR}"
            "$<TARGET_FILE_DIR:${target}>/$<TARGET_FILE_BASE_NAME:${target}>.wasm"
        DEPENDS "${target}"
        VERBATIM)
endfunction()
