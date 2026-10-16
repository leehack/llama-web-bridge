# llama.cpp v0.7.0 changes the graph for unpooled embeddings too, but its
# extraction toggle only reserves pooled graphs (llamadart-native#114).
# Compile a corrected build-directory copy; never modify the source checkout.
set(LLAMADART_CONTEXT_SOURCE "${LLAMA_CPP_DIR}/src/llama-context.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LLAMADART_CONTEXT_SOURCE}")
file(READ "${LLAMADART_CONTEXT_SOURCE}" LLAMADART_CONTEXT_CONTENT)
set(LLAMADART_EMBEDDING_RESERVE_GUARD
    "if (cparams.embeddings != value && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE)")
string(FIND "${LLAMADART_CONTEXT_CONTENT}" "${LLAMADART_EMBEDDING_RESERVE_GUARD}"
    LLAMADART_EMBEDDING_RESERVE_INDEX)
if (NOT LLAMADART_EMBEDDING_RESERVE_INDEX EQUAL -1)
    file(SHA256 "${LLAMADART_CONTEXT_SOURCE}" LLAMADART_CONTEXT_SHA256)
    if (NOT LLAMADART_CONTEXT_SHA256 STREQUAL
        "2a25e556cb0d16725e01ce853fb1b76da91dcb41b803ce1a6bacec89bfb94aa5")
        message(FATAL_ERROR "Unaudited embedding-mode reserve guard; requalify llama-context.cpp")
    endif()
    string(REPLACE "${LLAMADART_EMBEDDING_RESERVE_GUARD}"
        "if (cparams.embeddings != value)"
        LLAMADART_CONTEXT_CONTENT "${LLAMADART_CONTEXT_CONTENT}")
    set(LLAMADART_CONTEXT_PATCHED "${CMAKE_BINARY_DIR}/generated/llama-context-embedding-reserve.cpp")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
    file(WRITE "${LLAMADART_CONTEXT_PATCHED}" "${LLAMADART_CONTEXT_CONTENT}")
    set_source_files_properties("${LLAMADART_CONTEXT_SOURCE}"
        DIRECTORY "${LLAMA_CPP_DIR}/src" PROPERTIES HEADER_FILE_ONLY TRUE)
    target_sources(llama PRIVATE "${LLAMADART_CONTEXT_PATCHED}")
    set_source_files_properties("${LLAMADART_CONTEXT_PATCHED}"
        DIRECTORY "${LLAMA_CPP_DIR}/src" PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()
