#include "llama_webgpu_fetch_fs.h"

#include <cerrno>

#if defined(__EMSCRIPTEN_PTHREADS__)
#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>

#include <emscripten/threading.h>
#include <emscripten/wasmfs.h>

#include "backend.h"
#include "file.h"
#include "memory_backend.h"
#include "thread_utils.h"
#include "wasmfs.h"

extern "C" {
void _llamadart_fetch_fs_alloc_file(
    em_proxying_ctx * ctx, uint32_t file_id, const char * url, uint32_t chunk_size);
void _llamadart_fetch_fs_free_file(em_proxying_ctx * ctx, uint32_t file_id);
void _llamadart_fetch_fs_read(
    em_proxying_ctx * ctx, uint32_t file_id, uint8_t * buffer,
    uint32_t length, int64_t offset, int32_t * result);
void _llamadart_fetch_fs_get_size(
    em_proxying_ctx * ctx, uint32_t file_id, int64_t * out_size);
}

namespace {

constexpr uint32_t kDefaultChunkSize = 16 * 1024 * 1024;

template <typename Size, typename ReadStep>
std::make_signed_t<Size> read_fetch_file(Size length, int64_t offset, ReadStep read_step) {
  using SignedSize = std::make_signed_t<Size>;
  if (offset < 0) {
    return -EINVAL;
  }
  // WasmFS mmap zero-fills short reads, so an unrepresentable count must fail.
  if (length > static_cast<Size>(std::numeric_limits<SignedSize>::max())) {
    return -EOVERFLOW;
  }

  constexpr Size max_step = 1u << 30;
  const auto offset_headroom = static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - offset);
  Size total = 0;
  while (total < length) {
    if (static_cast<uint64_t>(total) > offset_headroom) {
      return -EOVERFLOW;
    }
    const auto step = static_cast<uint32_t>(std::min(length - total, max_step));
    const int64_t step_offset = offset + static_cast<int64_t>(total);
    const int32_t got = read_step(total, step, step_offset);
    if (got < 0) {
      return got;
    }
    if (got == 0) {
      break;
    }
    if (static_cast<uint32_t>(got) > step) {
      return -EIO;
    }
    total += static_cast<Size>(got);
  }
  return static_cast<SignedSize>(total);
}

class SharedFetchFile : public wasmfs::DataFile {
  emscripten::ProxyWorker & proxy_;
  const uint32_t file_id_;

public:
  SharedFetchFile(
      mode_t mode, wasmfs::backend_t backend, emscripten::ProxyWorker & proxy,
      uint32_t file_id, const std::string & url, uint32_t chunk_size)
      : wasmfs::DataFile(mode, backend), proxy_(proxy), file_id_(file_id) {
    proxy_([&](auto ctx) {
      _llamadart_fetch_fs_alloc_file(ctx.ctx, file_id_, url.c_str(), chunk_size);
    });
  }

  ~SharedFetchFile() override {
    proxy_([&](auto ctx) { _llamadart_fetch_fs_free_file(ctx.ctx, file_id_); });
  }

  int open(wasmfs::oflags_t) override { return 0; }
  int close() override { return 0; }
  int flush() override { return 0; }
  ssize_t write(const uint8_t *, size_t, off_t) override { return -EROFS; }
  int setSize(off_t) override { return -EROFS; }

  ssize_t read(uint8_t * buffer, size_t length, off_t offset) override {
    static_assert(std::numeric_limits<ssize_t>::max() ==
                  std::numeric_limits<std::make_signed_t<size_t>>::max());
    return read_fetch_file(length, offset, [&](size_t total, uint32_t step, int64_t step_offset) {
      int32_t result = -EBADF;
      proxy_([&](auto ctx) {
        _llamadart_fetch_fs_read(
            ctx.ctx, file_id_, buffer + total, step, step_offset, &result);
      });
      return result;
    });
  }

  off_t getSize() override {
    int64_t size = 0;
    proxy_([&](auto ctx) { _llamadart_fetch_fs_get_size(ctx.ctx, file_id_, &size); });
    return static_cast<off_t>(size);
  }
};

class SharedFetchBackend : public wasmfs::Backend {
  emscripten::ProxyWorker proxy_;
  std::mutex create_mutex_;
  std::string pending_url_;
  uint32_t pending_chunk_size_ = 0;
  uint32_t next_file_id_ = 0;

public:
  int create_mounted_file(const char * path, const char * url, uint32_t chunk_size) {
    std::lock_guard lock(create_mutex_);
    if (next_file_id_ == std::numeric_limits<uint32_t>::max()) {
      return -EOVERFLOW;
    }

    // WasmFS invokes createFile synchronously and skips it on a path collision.
    struct ResetPending {
      SharedFetchBackend & backend;
      ~ResetPending() {
        backend.pending_url_.clear();
        backend.pending_chunk_size_ = 0;
      }
    } reset{*this};
    pending_url_ = url;
    pending_chunk_size_ = chunk_size == 0 ? kDefaultChunkSize : chunk_size;
    return wasmfs_create_file(path, 0444, reinterpret_cast<::backend_t>(
        static_cast<wasmfs::Backend *>(this)));
  }

  std::shared_ptr<wasmfs::DataFile> createFile(mode_t mode) override {
    return std::make_shared<SharedFetchFile>(
        mode, this, proxy_, ++next_file_id_, pending_url_, pending_chunk_size_);
  }

  std::shared_ptr<wasmfs::Directory> createDirectory(mode_t mode) override {
    return std::make_shared<wasmfs::MemoryDirectory>(mode, this);
  }

  std::shared_ptr<wasmfs::Symlink> createSymlink(std::string target) override {
    return std::make_shared<wasmfs::MemorySymlink>(target, this);
  }
};

SharedFetchBackend & get_or_create_shared_fetch_backend() {
  assert(!emscripten_is_main_browser_thread());
  static std::mutex mutex;
  static SharedFetchBackend * backend = nullptr;
  std::lock_guard lock(mutex);
  if (backend == nullptr) {
    // WasmFS retains backends for the life of the core, including their worker.
    backend = static_cast<SharedFetchBackend *>(
        wasmfs::wasmFS.addBackend(std::make_unique<SharedFetchBackend>()));
  }
  return *backend;
}

}  // namespace
#endif

namespace llamadart_webgpu {

#if defined(__EMSCRIPTEN_PTHREADS__)
int create_shared_fetch_file(const char * path, const char * model_url, uint32_t chunk_size) {
  return get_or_create_shared_fetch_backend().create_mounted_file(path, model_url, chunk_size);
}
#else
int create_shared_fetch_file(const char *, const char *, uint32_t) {
  return -ENOSYS;
}
#endif

}  // namespace llamadart_webgpu
