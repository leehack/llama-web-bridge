#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

#include <emscripten/emscripten.h>
#include <emscripten/heap.h>
#include <emscripten/wasmfs.h>

#include "backend.h"
#include "file.h"
#include "llama_webgpu_heap_model_blocks.h"
#include "memory_backend.h"
#include "wasmfs.h"

extern "C" {
void _llamadart_js_model_fs_set_next_prealloc_bytes(int64_t bytes);
void _llamadart_js_model_fs_alloc_file(
    uintptr_t file_id,
    int32_t mount_id,
    uint8_t * heap_ptr,
    size_t heap_capacity);
void * __real_mmap(void * addr, size_t len, int prot, int flags, int fd, off_t off);
int __real_munmap(void * addr, size_t len);
void _llamadart_js_model_fs_free_file(uintptr_t file_id);
int32_t _llamadart_js_model_fs_bind_direct(uintptr_t file_id);
void _llamadart_js_model_fs_unbind_direct(int32_t handle);
int32_t _llamadart_js_model_fs_write(
    uintptr_t file_id,
    const uint8_t * buffer,
    uint32_t length,
    int64_t offset);
int32_t _llamadart_js_model_fs_read(
    uintptr_t file_id,
    uint8_t * buffer,
    uint32_t length,
    int64_t offset);
int32_t _llamadart_js_model_fs_read_async(
    uintptr_t file_id,
    uint8_t * buffer,
    uint32_t length,
    int64_t offset);
void _llamadart_js_model_fs_mark_external(uintptr_t file_id, double size, int32_t mode);
void _llamadart_js_model_fs_get_size(uintptr_t file_id, int64_t * out_size);
int32_t _llamadart_js_model_fs_set_size(uintptr_t file_id, int64_t size);
void _llamadart_js_model_fs_release_mount(int32_t mount_id);
double _llamadart_js_model_fs_consume_oom_bytes();
double _llamadart_js_model_fs_allocated_bytes();
}

namespace {

constexpr bool kHeapModelStagingEnabled = LLAMADART_WEBGPU_ALLOW_MEMORY_GROWTH != 0;
thread_local uintptr_t g_last_opened_file_id = 0;
// Set by open_direct_heap for the file its open() creates.
thread_local size_t g_next_heap_bytes = 0;
thread_local bool g_next_heap_failed = false;
thread_local int32_t g_next_external_mode = -1;
constexpr size_t kExternalReadBytes = 16 * 1024 * 1024;

llamadart_webgpu::HeapModelBlocks & heap_model_blocks() {
  static auto * blocks = new llamadart_webgpu::HeapModelBlocks();
  return *blocks;
}

size_t heap_model_allocation_limit() {
  constexpr size_t allocator_slack = 2 * llamadart_webgpu::HeapModelBlocks::kAlign;
  const size_t maximum = emscripten_get_heap_max();
  return maximum > allocator_slack ? maximum - allocator_slack : 0;
}

void release_js_model_mount(int32_t mount_id) {
  _llamadart_js_model_fs_release_mount(mount_id);
  if constexpr (kHeapModelStagingEnabled) {
    heap_model_blocks().on_mount_released(mount_id);
  }
}

class ChunkedJsModelFile : public wasmfs::DataFile {
  const int32_t external_mode_ = g_next_external_mode;

  uintptr_t getFileIndex() const {
    return reinterpret_cast<uintptr_t>(this);
  }

  int open(wasmfs::oflags_t) override {
    g_last_opened_file_id = getFileIndex();
    return 0;
  }
  int close() override { return 0; }
  int flush() override { return 0; }

  ssize_t write(const uint8_t * buf, size_t len, off_t offset) override {
    const size_t max_chunk = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    const uint32_t clamped = static_cast<uint32_t>(len > max_chunk ? max_chunk : len);
    return _llamadart_js_model_fs_write(
        getFileIndex(),
        buf,
        clamped,
        static_cast<int64_t>(offset));
  }

  ssize_t read(uint8_t * buf, size_t len, off_t offset) override {
    if (offset < 0) {
      return -EINVAL;
    }
    const size_t max_chunk = external_mode_ >= 0
        ? kExternalReadBytes
        : static_cast<size_t>(std::numeric_limits<int32_t>::max());
    const size_t readable = std::min(len, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
    const auto offset_headroom = static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - offset);
    size_t total = 0;
    // A JS call returns i32, but WASMFS mmap zero-fills any short memory64 read.
    while (total < readable) {
      if (total > offset_headroom) {
        return -EOVERFLOW;
      }
      const auto step = static_cast<uint32_t>(std::min(readable - total, max_chunk));
      const int64_t read_offset = offset + static_cast<int64_t>(total);
      const int32_t got = external_mode_ == 1
          ? _llamadart_js_model_fs_read_async(getFileIndex(), buf + total, step, read_offset)
          : _llamadart_js_model_fs_read(getFileIndex(), buf + total, step, read_offset);
      if (got < 0) {
        // A short count would let a WASMFS mmap zero-fill the rest of an
        // external file whose size is known.
        return external_mode_ >= 0 || total == 0 ? got : static_cast<ssize_t>(total);
      }
      total += static_cast<size_t>(got);
      if (static_cast<uint32_t>(got) < step) {
        break;
      }
    }
    return static_cast<ssize_t>(total);
  }

  off_t getSize() override {
    int64_t size = 0;
    _llamadart_js_model_fs_get_size(getFileIndex(), &size);
    return static_cast<off_t>(size);
  }

  int setSize(off_t size) override {
    return _llamadart_js_model_fs_set_size(
        getFileIndex(),
        static_cast<int64_t>(size));
  }

public:
  ChunkedJsModelFile(mode_t mode, wasmfs::backend_t backend, int32_t mount_id)
      : wasmfs::DataFile(mode, backend) {
    uint8_t * heap_ptr = nullptr;
    size_t heap_capacity = 0;
    if constexpr (kHeapModelStagingEnabled) {
      const size_t wanted = g_next_heap_bytes;
      g_next_heap_bytes = 0;
      if (wanted > 0) {
        heap_ptr = heap_model_blocks().create(
            getFileIndex(), mount_id, wanted, &heap_capacity, heap_model_allocation_limit());
        g_next_heap_failed = heap_ptr == nullptr;
      }
    }
    _llamadart_js_model_fs_alloc_file(getFileIndex(), mount_id, heap_ptr, heap_capacity);
  }

  ~ChunkedJsModelFile() override {
    _llamadart_js_model_fs_free_file(getFileIndex());
    if constexpr (kHeapModelStagingEnabled) {
      heap_model_blocks().on_file_destroyed(getFileIndex());
    }
  }
};

class ChunkedJsModelBackend : public wasmfs::Backend {
  int32_t mount_id_ = 0;

public:
  explicit ChunkedJsModelBackend(int32_t mount_id) : mount_id_(mount_id) {}

  std::shared_ptr<wasmfs::DataFile> createFile(mode_t mode) override {
    return std::make_shared<ChunkedJsModelFile>(mode, this, mount_id_);
  }

  std::shared_ptr<wasmfs::Directory> createDirectory(mode_t mode) override {
    return std::make_shared<wasmfs::MemoryDirectory>(mode, this);
  }

  std::shared_ptr<wasmfs::Symlink> createSymlink(std::string target) override {
    return std::make_shared<wasmfs::MemorySymlink>(target, this);
  }
};

struct MountedJsModelEntry {
  int32_t mount_id = 0;
  wasmfs::backend_t backend = nullptr;
};

bool g_has_js_model_mounts = false;

std::unordered_map<std::string, MountedJsModelEntry> & mounted_js_model_paths() {
  static auto * paths = new std::unordered_map<std::string, MountedJsModelEntry>();
  return *paths;
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_js_model_fs_api_version() {
  return 5;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_js_model_fs_mount_id(const char * mount_path) {
  if (mount_path == nullptr || std::strlen(mount_path) == 0) {
    return 0;
  }
  const auto & mounted = mounted_js_model_paths();
  const auto it = mounted.find(std::string(mount_path));
  if (it == mounted.end()) {
    return 0;
  }
  const auto current_backend =
      reinterpret_cast<wasmfs::backend_t>(wasmfs_get_backend_by_path(mount_path));
  if (current_backend == nullptr || current_backend != it->second.backend) {
    return 0;
  }
  return it->second.mount_id;
}

EMSCRIPTEN_KEEPALIVE void llamadart_webgpu_js_model_fs_release_mount_chunks(int32_t mount_id) {
  if (mount_id > 0) {
    release_js_model_mount(mount_id);
  }
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_mount_js_model_fs(const char * mount_path) {
  if (mount_path == nullptr || std::strlen(mount_path) == 0) {
    return -EINVAL;
  }

  std::string path_str(mount_path);
  auto & mounted = mounted_js_model_paths();
  if (wasmfs_get_backend_by_path(mount_path) != nullptr ||
      mounted.find(path_str) != mounted.end()) {
    return -EEXIST;
  }

  static int32_t next_mount_id = 1;
  int32_t mount_id = next_mount_id++;
  if (next_mount_id <= 0) {
    next_mount_id = 1;
  }

  // Allocate one fresh backend per mount rather than recycling backends, because
  // WasmFS retains created backends and cwd handles across unmounts.
  wasmfs::backend_t backend =
      wasmfs::wasmFS.addBackend(std::make_unique<ChunkedJsModelBackend>(mount_id));
  if (backend == nullptr) {
    return -ENOMEM;
  }

  const int rc = wasmfs_create_directory(
      mount_path,
      0777,
      reinterpret_cast<::backend_t>(backend));
  if (rc != 0) {
    return rc;
  }

  mounted.emplace(std::move(path_str), MountedJsModelEntry{mount_id, backend});
  g_has_js_model_mounts = true;
  return 0;
}

bool llamadart_webgpu_is_js_model_path(const char * path) {
  // Fast zero-allocation check so staging-off builds never construct the map.
  if (!g_has_js_model_mounts || path == nullptr || std::strlen(path) == 0) {
    return false;
  }
  const auto & mounted = mounted_js_model_paths();
  if (mounted.empty()) {
    return false;
  }
  std::string path_str(path);
  for (const auto & entry : mounted) {
    const std::string & mount = entry.first;
    if (path_str == mount) {
      return true;
    }
    if (path_str.size() > mount.size() &&
        path_str.compare(0, mount.size(), mount) == 0 &&
        path_str[mount.size()] == '/') {
      return true;
    }
  }
  return false;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_unmount_js_model_fs(const char * mount_path) {
  auto & mounted = mounted_js_model_paths();

  if (mount_path == nullptr || std::strlen(mount_path) == 0) {
    int first_error = 0;
    for (auto it = mounted.begin(); it != mounted.end(); ) {
      const int rc = wasmfs_unmount(it->first.c_str());
      if (rc != 0 && rc != -ENOENT) {
        if (first_error == 0) {
          first_error = rc;
        }
        ++it;
        continue;
      }
      release_js_model_mount(it->second.mount_id);
      it = mounted.erase(it);
    }
    g_has_js_model_mounts = !mounted.empty();
    return first_error;
  }

  auto it = mounted.find(std::string(mount_path));
  if (it == mounted.end()) {
    return wasmfs_get_backend_by_path(mount_path) != nullptr ? -EINVAL : 0;
  }

  const int rc = wasmfs_unmount(mount_path);
  if (rc != 0 && rc != -ENOENT) {
    return rc;
  }

  release_js_model_mount(it->second.mount_id);
  mounted.erase(it);
  g_has_js_model_mounts = !mounted.empty();
  return 0;
}

EMSCRIPTEN_KEEPALIVE void llamadart_webgpu_js_model_fs_prealloc_next(double bytes) {
  if (!(bytes > 0.0)) {
    _llamadart_js_model_fs_set_next_prealloc_bytes(0);
    return;
  }
  _llamadart_js_model_fs_set_next_prealloc_bytes(static_cast<int64_t>(bytes));
}

}  // extern "C"

namespace {

int32_t open_direct_file(
    const char * file_path,
    double prealloc_bytes,
    int32_t append,
    size_t heap_bytes) {
  if (file_path == nullptr || std::strlen(file_path) == 0 || !llamadart_webgpu_is_js_model_path(file_path)) {
    return -EINVAL;
  }
  if (append == 0) {
    if (::unlink(file_path) != 0 && errno != ENOENT) {
      return -(errno > 0 ? errno : EIO);
    }
    if (prealloc_bytes > 0.0) {
      _llamadart_js_model_fs_set_next_prealloc_bytes(static_cast<int64_t>(prealloc_bytes));
    } else {
      _llamadart_js_model_fs_set_next_prealloc_bytes(0);
    }
  } else {
    _llamadart_js_model_fs_set_next_prealloc_bytes(0);
  }

  g_next_heap_bytes = heap_bytes;
  g_next_heap_failed = false;
  g_last_opened_file_id = 0;
  const int flags = append != 0 ? (O_RDWR | O_CREAT) : (O_RDWR | O_CREAT | O_TRUNC);
  const int fd = ::open(file_path, flags, 0666);
  const int open_errno = errno;
  _llamadart_js_model_fs_set_next_prealloc_bytes(0);
  g_next_heap_bytes = 0;
  const bool heap_failed = g_next_heap_failed;
  g_next_heap_failed = false;
  if (fd < 0) {
    return -(open_errno > 0 ? open_errno : EIO);
  }

  const uintptr_t file_id = g_last_opened_file_id;
  g_last_opened_file_id = 0;
  ::close(fd);
  if (file_id == 0) {
    return -EBADF;
  }
  if constexpr (kHeapModelStagingEnabled) {
    if (heap_bytes > 0 && (heap_failed || !heap_model_blocks().is_live(file_id))) {
      ::unlink(file_path);
      return -ENOMEM;
    }
  }

  const int32_t handle = _llamadart_js_model_fs_bind_direct(file_id);
  if (handle <= 0) {
    if (append == 0) {
      ::unlink(file_path);
    }
    return handle < 0 ? handle : -EBADF;
  }
  return handle;
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_js_model_fs_open_direct(
    const char * file_path,
    double prealloc_bytes,
    int32_t append) {
  return open_direct_file(file_path, prealloc_bytes, append, 0);
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_js_model_fs_open_external(
    const char * file_path,
    double byte_length,
    int32_t mode) {
  constexpr double max_exact_js_size = 9007199254740991.0;
  if (file_path == nullptr || !(byte_length > 0.0) ||
      byte_length > max_exact_js_size || byte_length > static_cast<double>(SIZE_MAX) ||
      std::trunc(byte_length) != byte_length || (mode != 0 && mode != 1)) {
    return -EINVAL;
  }

  // Only a direct child of a currently mounted private filesystem can consume
  // the pending external mode during open().
  const std::string path(file_path);
  const auto slash = path.rfind('/');
  const auto name = path.substr(slash == std::string::npos ? 0 : slash + 1);
  if (slash == std::string::npos || name.empty() || name == "." || name == ".." ||
      llamadart_webgpu_js_model_fs_mount_id(path.substr(0, slash).c_str()) <= 0) {
    return -EINVAL;
  }

  _llamadart_js_model_fs_set_next_prealloc_bytes(0);
  g_last_opened_file_id = 0;
  g_next_external_mode = mode;
  const int fd = ::open(file_path, O_RDONLY | O_CREAT | O_EXCL, 0444);
  const int open_errno = errno;
  g_next_external_mode = -1;
  const uintptr_t file_id = g_last_opened_file_id;
  g_last_opened_file_id = 0;
  if (fd < 0) {
    return -(open_errno > 0 ? open_errno : EIO);
  }
  ::close(fd);
  if (file_id == 0) {
    ::unlink(file_path);
    return -EBADF;
  }
  _llamadart_js_model_fs_mark_external(file_id, byte_length, mode);
  const int32_t handle = _llamadart_js_model_fs_bind_direct(file_id);
  if (handle <= 0) {
    ::unlink(file_path);
    return handle < 0 ? handle : -EBADF;
  }
  return handle;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_js_model_fs_supports_heap_resident() {
  return kHeapModelStagingEnabled;
}

// Like open_direct(path, total, 0), but the new file lives in one wasm heap
// block of total_bytes that llama.cpp memory-maps without copying.
EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_js_model_fs_open_direct_heap(
    const char * file_path,
    double total_bytes) {
  if constexpr (!kHeapModelStagingEnabled) {
    return -ENOSYS;
  }
  if (!(total_bytes > 0.0) || total_bytes > static_cast<double>(SIZE_MAX - llamadart_webgpu::HeapModelBlocks::kAlign)) {
    return -EINVAL;
  }
  return open_direct_file(file_path, 0.0, 0, static_cast<size_t>(total_bytes));
}

EMSCRIPTEN_KEEPALIVE double llamadart_webgpu_js_model_fs_heap_resident_bytes() {
  return kHeapModelStagingEnabled ? static_cast<double>(heap_model_blocks().resident_bytes()) : 0;
}

bool llamadart_webgpu_is_heap_model_path(const char * path) {
  if (!kHeapModelStagingEnabled || !llamadart_webgpu_is_js_model_path(path)) {
    return false;
  }
  struct stat st;
  return ::stat(path, &st) == 0 && heap_model_blocks().is_live(static_cast<uintptr_t>(st.st_ino));
}

// Every llama.cpp object that mapped a staged file must be destroyed first.
void llamadart_webgpu_release_heap_model_mappings() {
  if constexpr (kHeapModelStagingEnabled) {
    heap_model_blocks().release_mappings();
  }
}

#if LLAMADART_WEBGPU_ALLOW_MEMORY_GROWTH
void * __wrap_mmap(void * addr, size_t len, int prot, int flags, int fd, off_t off) {
  if (fd >= 0 && g_has_js_model_mounts) {
    struct stat st;
    if (::fstat(fd, &st) == 0 && st.st_size >= 0) {
      void * mapped = heap_model_blocks().map(
          static_cast<uintptr_t>(st.st_ino),
          static_cast<size_t>(st.st_size),
          len,
          static_cast<int64_t>(off),
          (prot & PROT_WRITE) != 0,
          (flags & MAP_FIXED) != 0);
      if (mapped != nullptr) {
        return mapped;
      }
    }
  }
  return __real_mmap(addr, len, prot, flags, fd, off);
}

// llama.cpp unmaps fragments of a mapping; the block is freed as a whole by
// release_mappings().
int __wrap_munmap(void * addr, size_t len) {
  if (heap_model_blocks().contains(addr)) {
    return 0;
  }
  return __real_munmap(addr, len);
}
#endif

EMSCRIPTEN_KEEPALIVE void llamadart_webgpu_js_model_fs_close_direct(int32_t handle) {
  _llamadart_js_model_fs_unbind_direct(handle);
}

EMSCRIPTEN_KEEPALIVE double llamadart_webgpu_js_model_fs_consume_oom_bytes() {
  return _llamadart_js_model_fs_consume_oom_bytes();
}

EMSCRIPTEN_KEEPALIVE double llamadart_webgpu_js_model_fs_allocated_bytes() {
  return _llamadart_js_model_fs_allocated_bytes();
}

}  // extern "C"
