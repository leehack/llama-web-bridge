export const heapBlocksHarness = String.raw`
#include "llama_webgpu_heap_model_blocks.h"

#include <cassert>
#include <vector>

using llamadart_webgpu::HeapModelBlocks;

static std::vector<void *> g_released;
static void counted_free(void * ptr) {
  g_released.push_back(ptr);
  std::free(ptr);
}

int main() {
  constexpr size_t kAlign = HeapModelBlocks::kAlign;
  HeapModelBlocks blocks(counted_free);
  size_t capacity = 0;
  assert(blocks.create(0, 10, kAlign + 1, &capacity, kAlign + 1) == nullptr);
  assert(capacity == 0 && blocks.resident_bytes() == 0);

  uint8_t * a = blocks.create(1, 10, kAlign + 5, &capacity);
  assert(a != nullptr && reinterpret_cast<uintptr_t>(a) % kAlign == 0);
  assert(capacity == 2 * kAlign && blocks.resident_bytes() == 2 * kAlign);
  assert(blocks.create(2, 10, 0, &capacity) == nullptr);
  std::memset(a, 0x5a, 2 * kAlign);

  const size_t size = kAlign + 5;
  assert(blocks.map(99, size, size, 0, false, false) == nullptr);
  assert(blocks.map(1, size, size, 4096, false, false) == nullptr);
  assert(blocks.map(1, size, size, 0, true, false) == nullptr);
  assert(blocks.map(1, size, size, 0, false, true) == nullptr);
  assert(blocks.map(1, size, 2 * kAlign + 1, 0, false, false) == nullptr);
  assert(blocks.map(1, size, 0, 0, false, false) == nullptr);
  assert(a[size] == 0x5a);

  assert(blocks.map(1, size, size, 0, false, false) == a);
  assert(a[size - 1] == 0x5a && a[size] == 0 && a[2 * kAlign - 1] == 0);
  std::memset(a, 0x5a, 2 * kAlign);
  assert(blocks.map(1, 5, 5, 0, false, false) == a);
  assert(a[4] == 0x5a && a[5] == 0 && a[kAlign - 1] == 0 && a[kAlign] == 0x5a);

  assert(blocks.contains(a) && blocks.contains(a + 2 * kAlign - 1));
  assert(!blocks.contains(a + 2 * kAlign) && !blocks.contains(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(a) - 1)));

  a[0] = 0x77;
  blocks.on_file_destroyed(1);
  assert(!blocks.is_live(1) && blocks.resident_bytes() == 2 * kAlign);
  assert(g_released.empty());
  assert(a[0] == 0x77 && blocks.contains(a));
  assert(blocks.map(1, size, size, 0, false, false) == nullptr);

  uint8_t * b = blocks.create(1, 11, 10, &capacity);
  assert(b != nullptr && b != a && blocks.resident_bytes() == 3 * kAlign);
  assert(a[0] == 0x77);

  blocks.release_mappings();
  assert(blocks.resident_bytes() == kAlign && !blocks.contains(a) && blocks.contains(b));
  assert(g_released == std::vector<void *>{a});

  blocks.on_file_destroyed(1);
  assert(blocks.resident_bytes() == 0);
  assert((g_released == std::vector<void *>{a, b}));
  uint8_t * c = blocks.create(3, 20, 10, &capacity);
  uint8_t * d = blocks.create(4, 21, 10, &capacity);
  assert(c != nullptr && d != nullptr);
  assert(blocks.map(3, 10, 10, 0, false, false) == c);
  blocks.on_mount_released(21);
  assert(blocks.resident_bytes() == kAlign && blocks.is_live(3) && !blocks.is_live(4));
  blocks.on_mount_released(20);
  assert(blocks.resident_bytes() == kAlign && !blocks.is_live(3));
  blocks.on_file_destroyed(3);
  blocks.release_mappings();
  assert(blocks.resident_bytes() == 0);
  assert((g_released == std::vector<void *>{a, b, d, c}));

  uint8_t * e = blocks.create(5, 30, 10, &capacity);
  assert(blocks.map(5, 10, 10, 0, false, false) == e);
  blocks.release_mappings();
  assert(blocks.resident_bytes() == kAlign && blocks.is_live(5));
  assert(g_released.size() == 4);
  blocks.on_file_destroyed(5);
  assert(blocks.resident_bytes() == 0);
  assert(g_released.size() == 5 && g_released.back() == e);

  return 0;
}
`;
