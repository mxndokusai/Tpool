#pragma once

#include "mpmc.hpp"
#include <atomic>
#include <concepts>
#include <coroutine>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <semaphore>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

template <typename T> class task;

template <typename T> struct task_promise {
  struct final_awaitable {
    bool await_ready() const noexcept { return false; }

    std::coroutine_handle<>
    await_suspend(std::coroutine_handle<task_promise> h) noexcept {
      if (h.promise().continuation_) {
        return h.promise().continuation_;
      }
      return std::noop_coroutine();
    }

    void await_resume() noexcept {}
  };

  task<T> get_return_object() noexcept;

  std::suspend_always initial_suspend() noexcept { return {}; }
  final_awaitable final_suspend() noexcept { return {}; }

  // Only return_value for non-void T.
  void return_value(T value) { result_ = std::move(value); }

  void unhandled_exception() { result_ = std::current_exception(); }

  std::variant<std::monostate, T, std::exception_ptr> result_;
  std::coroutine_handle<> continuation_;
};

// Template specialisation for void.
template <> struct task_promise<void> {
  struct final_awaitable {
    bool await_ready() const noexcept { return false; }

    std::coroutine_handle<>
    await_suspend(std::coroutine_handle<task_promise> h) noexcept {
      if (h.promise().continuation_) {
        return h.promise().continuation_;
      }
      return std::noop_coroutine();
    }

    void await_resume() noexcept {}
  };

  task<void> get_return_object() noexcept;

  std::suspend_always initial_suspend() noexcept { return {}; }
  final_awaitable final_suspend() noexcept { return {}; }

  // Only return_void for void.
  void return_void() noexcept {}

  void unhandled_exception() { exception_ = std::current_exception(); }

  std::exception_ptr exception_;
  std::coroutine_handle<> continuation_;
};

// Non-void task.
template <typename T> class [[nodiscard]] task {
public:
  using promise_type = task_promise<T>;
  using handle_type = std::coroutine_handle<promise_type>;

  explicit task(handle_type handle) noexcept : handle_(handle) {}

  task(task &&other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}

  task &operator=(task &&other) noexcept {
    if (this != &other) {
      if (handle_)
        handle_.destroy();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  ~task() {
    if (handle_)
      handle_.destroy();
  }

  task(const task &) = delete;
  task &operator=(const task &) = delete;

  // Awaitable interface.
  bool await_ready() const noexcept { return false; }

  std::coroutine_handle<>
  await_suspend(std::coroutine_handle<> continuation) noexcept {
    handle_.promise().continuation_ = continuation;
    return handle_;
  }

  T await_resume() {
    auto &result = handle_.promise().result_;
    if (std::holds_alternative<std::exception_ptr>(result)) {
      std::rethrow_exception(std::get<std::exception_ptr>(result));
    }
    return std::get<T>(std::move(result));
  }

  void resume() {
    if (handle_ && !handle_.done()) {
      handle_.resume();
    }
  }

  bool done() const { return !handle_ || handle_.done(); }

private:
  handle_type handle_;
};

// Void task.
template <> class [[nodiscard]] task<void> {
public:
  using promise_type = task_promise<void>;
  using handle_type = std::coroutine_handle<promise_type>;

  explicit task(handle_type handle) noexcept : handle_(handle) {}

  task(task &&other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}

  task &operator=(task &&other) noexcept {
    if (this != &other) {
      if (handle_)
        handle_.destroy();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  ~task() {
    if (handle_)
      handle_.destroy();
  }

  task(const task &) = delete;
  task &operator=(const task &) = delete;

  // Awaitable interface.
  bool await_ready() const noexcept { return false; }

  std::coroutine_handle<>
  await_suspend(std::coroutine_handle<> continuation) noexcept {
    handle_.promise().continuation_ = continuation;
    return handle_;
  }

  void await_resume() {
    if (handle_.promise().exception_) {
      std::rethrow_exception(handle_.promise().exception_);
    }
  }

  void resume() {
    if (handle_ && !handle_.done()) {
      handle_.resume();
    }
  }

  bool done() const { return !handle_ || handle_.done(); }

private:
  handle_type handle_;
};

// Get return object implementations.
template <typename T>
inline task<T> task_promise<T>::get_return_object() noexcept {
  return task<T>{std::coroutine_handle<task_promise<T>>::from_promise(*this)};
}

inline task<void> task_promise<void>::get_return_object() noexcept {
  return task<void>{
      std::coroutine_handle<task_promise<void>>::from_promise(*this)};
}

// Execution context using mpmc.
class context {
public:
  explicit context(size_t num_threads = std::thread::hardware_concurrency(),
                   size_t queue_capacity = 1024)
      : work_queue_(queue_capacity), work_count_(0), stop_requested_(false) {

    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back(
          [this](std::stop_token stoken) { worker_loop(stoken); });
    }
  }

  ~context() { stop(); }

  // Non-copyable and non-movable.
  context(const context &) = delete;
  context &operator=(const context &) = delete;
  context(context &&) = delete;
  context &operator=(context &&) = delete;

  // Post work to the context, non-blocking if there is space.
  template <std::invocable Fn> bool try_post(Fn &&fn) {
    if (stop_requested_.load(std::memory_order_acquire)) {
      return false;
    }

    work_count_.fetch_add(1, std::memory_order_release);

    // Try emplace for non-blocking insertion into mpmc.
    bool success = work_queue_.try_emplace(std::forward<Fn>(fn));

    if (success) {
      work_available_.release();
    } else {
      work_count_.fetch_sub(1, std::memory_order_release);
    }

    return success;
  }

  // Post, blocks and waits if queue is full.
  template <std::invocable Fn> void post(Fn &&fn) {
    if (stop_requested_.load(std::memory_order_acquire)) {
      throw std::runtime_error("Cannot post to stopped execution context");
    }

    work_count_.fetch_add(1, std::memory_order_release);

    // Use emplace which blocks until space is available.
    work_queue_.emplace(std::forward<Fn>(fn));
    work_available_.release();
  }

  // Dispatch work, executes immediately if on worker thread, otherwise post.
  template <std::invocable Fn> void dispatch(Fn &&fn) {
    if (is_worker_thread()) {
      std::invoke(std::forward<Fn>(fn));
    } else {
      post(std::forward<Fn>(fn));
    }
  }

  // Schedule coroutine to run on this context.
  template <typename T> task<T> spawn(task<T> t) {
    struct awaiter {
      task<T> task_;
      context *ctx_;

      bool await_ready() const noexcept { return false; }

      void await_suspend(std::coroutine_handle<> continuation) {
        ctx_->post([this, continuation]() mutable {
          task_.resume();
          if (task_.done()) {
            continuation.resume();
          }
        });
      }

      T await_resume() { return task_.await_resume(); }
    };

    co_return co_await awaiter{std::move(t), this};
  }

  // Stop the context and join all threads.
  void stop() {
    bool expected = false;
    if (!stop_requested_.compare_exchange_strong(expected, true)) {
      return;
    }

    // Request stop for all threads.
    for (auto &worker : workers_) {
      worker.request_stop();
    }

    // Wake up all threads.
    for (size_t i = 0; i < workers_.size(); ++i) {
      work_available_.release();
    }

    // Join all threads.
    for (auto &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  // Wait for all pending work to complete.
  void wait() {
    while (work_count_.load(std::memory_order_acquire) > 0) {
      std::this_thread::yield();
    }
  }

  size_t thread_count() const { return workers_.size(); }

  bool stopped() const {
    return stop_requested_.load(std::memory_order_acquire);
  }

  // Get approximate queue size.
  ptrdiff_t queue_size() const { return work_queue_.size(); }

private:
  void worker_loop(std::stop_token stoken) {
    thread_local_worker_id_ = std::this_thread::get_id();

    while (!stoken.stop_requested()) {
      work_available_.acquire();

      if (stoken.stop_requested()) {
        break;
      }

      std::function<void()> task;

      // Use try_pop for non-blocking dequeue.
      if (work_queue_.try_pop(task)) {
        if (task) {
          try {
            task();
          } catch (...) {
            // TODO: Log/handle exception, add error handler callback?
          }
          work_count_.fetch_sub(1, std::memory_order_release);
        }
      }
    }
  }

  bool is_worker_thread() const {
    return thread_local_worker_id_ == std::this_thread::get_id();
  }

  std::vector<std::jthread> workers_;
  mpmc<std::function<void()>> work_queue_;
  std::counting_semaphore<> work_available_{0};
  std::atomic<size_t> work_count_;
  std::atomic<bool> stop_requested_;

  static thread_local std::thread::id thread_local_worker_id_;
};

thread_local std::thread::id context::thread_local_worker_id_;

// Coroutine-based thread pool.
class thread_pool {
public:
  explicit thread_pool(size_t num_threads = std::thread::hardware_concurrency(),
                       size_t queue_capacity = 1024)
      : context_(num_threads, queue_capacity) {}

  ~thread_pool() = default;
  // Non-copyable and non-movable.
  thread_pool(const thread_pool &) = delete;
  thread_pool &operator=(const thread_pool &) = delete;
  thread_pool(thread_pool &&) = delete;
  thread_pool &operator=(thread_pool &&) = delete;

  // Singleton accessor.
  static thread_pool &
  instance(size_t num_threads = std::thread::hardware_concurrency(),
           size_t queue_capacity = 1024) {
    static thread_pool pool(num_threads, queue_capacity);
    return pool;
  }

  // Submit callable that returns a future, non-blocking.
  template <typename Fn, typename... Args>
    requires std::invocable<Fn, Args...>
  std::optional<std::future<std::invoke_result_t<Fn, Args...>>>
  try_submit(Fn &&fn, Args &&...args) {
    using return_type = std::invoke_result_t<Fn, Args...>;
    using task_type = std::packaged_task<return_type()>;

    auto bound = std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...);
    auto task = std::make_shared<task_type>(std::move(bound));
    auto future = task->get_future();

    bool success =
        context_.try_post([task = std::move(task)]() mutable { (*task)(); });

    if (success)
      return future;
    else
      return std::nullopt;
  }

  // Submit callable that returns a future, blocking if full queue.
  template <typename Fn, typename... Args>
    requires std::invocable<Fn, Args...>
  auto submit(Fn &&fn, Args &&...args)
      -> std::future<std::invoke_result_t<Fn, Args...>> {
    using return_type = std::invoke_result_t<Fn, Args...>;
    using task_type = std::packaged_task<return_type()>;

    auto bound = std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...);
    auto task = std::make_shared<task_type>(std::move(bound));
    auto future = task->get_future();

    context_.post([task = std::move(task)]() mutable { (*task)(); });

    return future;
  }

  // Submit coroutine task.
  template <typename T> void submit(task<T> t) {
    context_.post([t = std::move(t)]() mutable { t.resume(); });
  }

  // Schedule coroutine on the pool.
  template <typename T> task<T> schedule(task<T> t) {
    return context_.spawn(std::move(t));
  }

  // Post work without getting a future back, non-blocking.
  template <std::invocable Fn> bool try_post(Fn &&fn) {
    return context_.try_post(std::forward<Fn>(fn));
  }

  // Post work without getting a future back, blocking.
  template <std::invocable Fn> void post(Fn &&fn) {
    context_.post(std::forward<Fn>(fn));
  }

  // Get the underlying execution context.
  context &get_context() { return context_; }
  const context &get_context() const { return context_; }

  // Stop the thread pool.
  void stop() { context_.stop(); }

  // Wait for all work to complete.
  void wait() { context_.wait(); }

  // Check if stopped.
  bool stopped() const { return context_.stopped(); }

  // Get thread count.
  size_t size() const { return context_.thread_count(); }

  // Get approximate queue size.
  ptrdiff_t queue_size() const { return context_.queue_size(); }

private:
  context context_;
};

// Awaitable that switches execution to thread pool.
struct tp_switch {
  thread_pool &pool_;

  bool await_ready() const noexcept { return false; }

  void await_suspend(std::coroutine_handle<> handle) {
    pool_.post([handle]() mutable { handle.resume(); });
  }

  void await_resume() const noexcept {}
};

// Helper function to switch to thread pool.
inline tp_switch resume_on(thread_pool &pool) { return tp_switch{pool}; }
