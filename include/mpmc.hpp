#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>

// Cache size alignment.
static constexpr std::size_t HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE = 64;

template <typename T> struct slot {
  ~slot() noexcept {
    if (turn & 1) {
      destroy();
    }
  }

  template <typename... Args> void construct(Args &&...args) noexcept {
    static_assert(std::is_nothrow_constructible<T, Args &&...>::value,
                  "T must be nothrow constructible with Args&&...");
    new (&storage) T(std::forward<Args>(args)...);
  }

  void destroy() noexcept {
    static_assert(std::is_nothrow_destructible<T>::value,
                  "T must be nothrow destructible");
    reinterpret_cast<T *>(&storage)->~T();
  }

  T &&move() noexcept { return reinterpret_cast<T &&>(storage); }

  // Align to avoid false sharing between adjacent slots.
  alignas(HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE)
      std::atomic<std::size_t> turn = {0};

  // C++20 replacement for std::aligned_storage.
  alignas(T) std::byte storage[sizeof(T)];
};

template <typename T, typename slot_allocator = std::allocator<slot<T>>>
class queue {
private:
  static_assert(std::is_nothrow_copy_assignable<T>::value ||
                    std::is_nothrow_move_assignable<T>::value,
                "T must be nothrow copy or move assignable");

  static_assert(std::is_nothrow_destructible<T>::value,
                "T must be nothrow destructible");

public:
  queue(const std::size_t capacity,
        const slot_allocator &allocator = slot_allocator())
      : capacity_(capacity), allocator_(allocator), head_(0), tail_(0) {
    if (capacity_ < 1) {
      throw std::invalid_argument("capacity < 1");
    }
    // Allocate one extra slot to prevent false sharing on the last slot.
    slots_ = allocator_.allocate(capacity_ + 1);
    // Not required to honour alignment for over-aligned types.
    if (reinterpret_cast<std::size_t>(slots_) % alignof(slot<T>) != 0) {
      allocator_.deallocate(slots_, capacity_ + 1);
      throw std::bad_alloc();
    }
    for (std::size_t i = 0; i < capacity_; ++i) {
      new (&slots_[i]) slot<T>();
    }
    static_assert(
        alignof(slot<T>) == HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE,
        "Slot must be aligned to cache line boundary to prevent false sharing");
    static_assert(sizeof(slot<T>) % HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE == 0,
                  "Slot size must be a multiple of cache line size to prevent "
                  "false sharing between adjacent slots");
    static_assert(sizeof(queue) % HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE == 0,
                  "Queue size must be a multiple of cache line size to "
                  "prevent false sharing between adjacent queues");
    static_assert(
        offsetof(queue, tail_) - offsetof(queue, head_) ==
            static_cast<std::ptrdiff_t>(HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE),
        "head and tail must be a cache line apart to prevent false sharing");
  }

  ~queue() noexcept {
    for (std::size_t i = 0; i < capacity_; ++i) {
      slots_[i].~slot();
    }
    allocator_.deallocate(slots_, capacity_ + 1);
  }

  // Non-copyable and non-movable.
  queue(const queue &) = delete;
  queue &operator=(const queue &) = delete;
  queue(queue &&) noexcept = delete;
  queue &operator=(queue &&) noexcept = delete;

  template <typename... Args> void emplace(Args &&...args) noexcept {
    static_assert(std::is_nothrow_constructible<T, Args &&...>::value,
                  "T must be nothrow constructible with Args&&...");
    auto const head = head_.fetch_add(1);
    auto &slot = slots_[idx(head)];
    while (turn(head) * 2 != slot.turn.load(std::memory_order_acquire))
      ;
    slot.construct(std::forward<Args>(args)...);
    slot.turn.store(turn(head) * 2 + 1, std::memory_order_release);
  }

  template <typename... Args> bool try_emplace(Args &&...args) noexcept {
    static_assert(std::is_nothrow_constructible<T, Args &&...>::value,
                  "T must be nothrow constructible with Args&&...");
    auto head = head_.load(std::memory_order_acquire);
    for (;;) {
      auto &slot = slots_[idx(head)];
      if (turn(head) * 2 == slot.turn.load(std::memory_order_acquire)) {
        if (head_.compare_exchange_strong(head, head + 1)) {
          slot.construct(std::forward<Args>(args)...);
          slot.turn.store(turn(head) * 2 + 1, std::memory_order_release);
          return true;
        }
      } else {
        auto const prevHead = head;
        head = head_.load(std::memory_order_acquire);
        if (head == prevHead) {
          return false;
        }
      }
    }
  }

  void push(const T &v) noexcept {
    static_assert(std::is_nothrow_copy_constructible<T>::value,
                  "T must be nothrow copy constructible");
    emplace(v);
  }

  template <typename P,
            typename = typename std::enable_if<
                std::is_nothrow_constructible<T, P &&>::value>::type>
  void push(P &&v) noexcept {
    emplace(std::forward<P>(v));
  }

  bool try_push(const T &v) noexcept {
    static_assert(std::is_nothrow_copy_constructible<T>::value,
                  "T must be nothrow copy constructible");
    return try_emplace(v);
  }

  template <typename P,
            typename = typename std::enable_if<
                std::is_nothrow_constructible<T, P &&>::value>::type>
  bool try_push(P &&v) noexcept {
    return try_emplace(std::forward<P>(v));
  }

  void pop(T &v) noexcept {
    auto const tail = tail_.fetch_add(1);
    auto &slot = slots_[idx(tail)];
    while (turn(tail) * 2 + 1 != slot.turn.load(std::memory_order_acquire))
      ;
    v = slot.move();
    slot.destroy();
    slot.turn.store(turn(tail) * 2 + 2, std::memory_order_release);
  }

  bool try_pop(T &v) noexcept {
    auto tail = tail_.load(std::memory_order_acquire);
    for (;;) {
      auto &slot = slots_[idx(tail)];
      if (turn(tail) * 2 + 1 == slot.turn.load(std::memory_order_acquire)) {
        if (tail_.compare_exchange_strong(tail, tail + 1)) {
          v = slot.move();
          slot.destroy();
          slot.turn.store(turn(tail) * 2 + 2, std::memory_order_release);
          return true;
        }
      } else {
        auto const prevTail = tail;
        tail = tail_.load(std::memory_order_acquire);
        if (tail == prevTail) {
          return false;
        }
      }
    }
  }

  // Returns the number of elements in the queue.
  // Size can be negative when queue is empty and at least one reader waiting.
  // Only best effort guess, there are read/write threads concurrently running.
  std::ptrdiff_t size() const noexcept {
    return static_cast<std::ptrdiff_t>(head_.load(std::memory_order_relaxed) -
                                       tail_.load(std::memory_order_relaxed));
  }

  // Returns true if the queue is empty.
  // Only best effort guess, there are read/write threads concurrently running.
  bool empty() const noexcept { return size() <= 0; }

private:
  constexpr std::size_t idx(std::size_t i) const noexcept {
    return i % capacity_;
  }

  constexpr std::size_t turn(std::size_t i) const noexcept {
    return i / capacity_;
  }

private:
  const std::size_t capacity_;
  slot<T> *slots_;
  // Optionally include [[no_unique_address]] attribute for allocator_.
  slot_allocator allocator_;

  // Align to avoid false sharing between head_ and tail_.
  alignas(
      HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE) std::atomic<std::size_t> head_;
  alignas(
      HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE) std::atomic<std::size_t> tail_;
};

template <typename T, typename slot_allocator = std::allocator<slot<T>>>
using mpmc = queue<T, slot_allocator>;
