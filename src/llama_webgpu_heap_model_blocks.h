#pragma once

// Ownership of heap-resident staged model files: each is one 64 KiB-aligned
// block that llama.cpp memory-maps without a copy. Dependency-free so the host
// contract test can compile it.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace llamadart_webgpu {

class HeapModelBlocks {
public:
  static constexpr size_t kAlign = 65536;

  struct Block {
    uint8_t * ptr = nullptr;
    size_t capacity = 0;
    int32_t mount_id = 0;
    bool pinned = false;
  };

  // The contract test passes its own release function to see each free.
  explicit HeapModelBlocks(void (*release)(void *) = std::free) : release_(release) {}

  // Returns the block for a new file, or nullptr when it cannot be allocated.
  uint8_t * create(
      uintptr_t file_id, int32_t mount_id, size_t bytes, size_t * capacity_out,
      size_t allocation_limit = SIZE_MAX) {
    if (bytes == 0 || bytes > SIZE_MAX - kAlign) {
      return nullptr;
    }
    const size_t capacity = (bytes + kAlign - 1) / kAlign * kAlign;
    if (capacity > allocation_limit) {
      return nullptr;
    }
    auto * ptr = static_cast<uint8_t *>(std::aligned_alloc(kAlign, capacity));
    if (ptr == nullptr) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    live_[file_id] = Block{ptr, capacity, mount_id, false};
    *capacity_out = capacity;
    return ptr;
  }

  // The file is gone. A mapped block outlives it until release_mappings().
  void on_file_destroyed(uintptr_t file_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    retire_locked(file_id);
  }

  void on_mount_released(int32_t mount_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uintptr_t> ids;
    for (const auto & entry : live_) {
      if (entry.second.mount_id == mount_id) {
        ids.push_back(entry.first);
      }
    }
    for (const uintptr_t id : ids) {
      retire_locked(id);
    }
  }

  bool is_live(uintptr_t file_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return live_.find(file_id) != live_.end();
  }

  // Returns the block when this mmap() request can be served from it, and pins
  // it. Bytes between the file size and the end of the last mapped page read as
  // zero, as POSIX requires, without touching the rest of the block.
  void * map(uintptr_t file_id, size_t file_size, size_t len, int64_t off, bool writable, bool fixed) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = live_.find(file_id);
    if (it == live_.end() || off != 0 || writable || fixed || len == 0) {
      return nullptr;
    }
    Block & block = it->second;
    if (len > block.capacity || file_size > block.capacity) {
      return nullptr;
    }
    const size_t mapped_end = (len + kAlign - 1) / kAlign * kAlign;
    if (file_size < mapped_end) {
      std::memset(block.ptr + file_size, 0, mapped_end - file_size);
    }
    block.pinned = true;
    return block.ptr;
  }

  bool contains(const void * addr) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto * target = static_cast<const uint8_t *>(addr);
    for (const auto & entry : live_) {
      if (target >= entry.second.ptr && target < entry.second.ptr + entry.second.capacity) {
        return true;
      }
    }
    for (const Block & block : orphans_) {
      if (target >= block.ptr && target < block.ptr + block.capacity) {
        return true;
      }
    }
    return false;
  }

  // Call only after every llama.cpp object that mapped a block is destroyed.
  void release_mappings() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & entry : live_) {
      entry.second.pinned = false;
    }
    for (const Block & block : orphans_) {
      release_(block.ptr);
    }
    orphans_.clear();
  }

  size_t resident_bytes() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto & entry : live_) {
      total += entry.second.capacity;
    }
    for (const Block & block : orphans_) {
      total += block.capacity;
    }
    return total;
  }

private:
  void retire_locked(uintptr_t file_id) {
    const auto it = live_.find(file_id);
    if (it == live_.end()) {
      return;
    }
    if (it->second.pinned) {
      orphans_.push_back(it->second);
    } else {
      release_(it->second.ptr);
    }
    live_.erase(it);
  }

  void (*release_)(void *);
  std::mutex mutex_;
  // Keyed by the WasmFS file address. A file is always destroyed, which moves
  // a mapped block to orphans_, before a later file can reuse its address.
  std::unordered_map<uintptr_t, Block> live_;
  std::vector<Block> orphans_;
};

}  // namespace llamadart_webgpu
