#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

#include <emscripten/emscripten.h>
#include <emscripten/wasmfs.h>

#include "backend.h"
#include "file.h"
#include "memory_backend.h"
#include "wasmfs.h"

extern "C" {
void _llamadart_js_model_fs_set_next_prealloc_bytes(int64_t bytes);
void _llamadart_js_model_fs_alloc_file(uintptr_t file_id, int32_t mount_id);
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
void _llamadart_js_model_fs_get_size(uintptr_t file_id, int64_t * out_size);
int32_t _llamadart_js_model_fs_set_size(uintptr_t file_id, int64_t size);
void _llamadart_js_model_fs_release_mount(int32_t mount_id);
double _llamadart_js_model_fs_consume_oom_bytes();
double _llamadart_js_model_fs_allocated_bytes();
}

namespace {

thread_local uintptr_t g_last_opened_file_id = 0;

class ChunkedJsModelFile : public wasmfs::DataFile {
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
    const size_t max_chunk = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    const uint32_t clamped = static_cast<uint32_t>(len > max_chunk ? max_chunk : len);
    return _llamadart_js_model_fs_read(
        getFileIndex(),
        buf,
        clamped,
        static_cast<int64_t>(offset));
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
    _llamadart_js_model_fs_alloc_file(getFileIndex(), mount_id);
  }

  ~ChunkedJsModelFile() override {
    _llamadart_js_model_fs_free_file(getFileIndex());
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
  return 3;
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
    _llamadart_js_model_fs_release_mount(mount_id);
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
      _llamadart_js_model_fs_release_mount(it->second.mount_id);
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

  _llamadart_js_model_fs_release_mount(it->second.mount_id);
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

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_js_model_fs_open_direct(
    const char * file_path,
    double prealloc_bytes,
    int32_t append) {
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

  g_last_opened_file_id = 0;
  const int flags = append != 0 ? (O_RDWR | O_CREAT) : (O_RDWR | O_CREAT | O_TRUNC);
  const int fd = ::open(file_path, flags, 0666);
  const int open_errno = errno;
  _llamadart_js_model_fs_set_next_prealloc_bytes(0);
  if (fd < 0) {
    return -(open_errno > 0 ? open_errno : EIO);
  }

  const uintptr_t file_id = g_last_opened_file_id;
  g_last_opened_file_id = 0;
  ::close(fd);
  if (file_id == 0) {
    return -EBADF;
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
