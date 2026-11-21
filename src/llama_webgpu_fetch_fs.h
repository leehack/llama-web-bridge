#pragma once

#include <cstdint>

namespace llamadart_webgpu {

int create_shared_fetch_file(
    const char * path, const char * model_url, uint32_t chunk_size);

}  // namespace llamadart_webgpu
