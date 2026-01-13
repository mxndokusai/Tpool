#include "pool.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace std::chrono_literals;

void benchmark_throughput() {
  std::cout << "=== Estimating Throughput ===" << std::endl;
  constexpr size_t NUM_TASKS = 1'000'000;
  constexpr size_t NUM_THREADS = 8;
  thread_pool pool(NUM_THREADS, 1024);
  std::atomic<size_t> completed{0};
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < NUM_TASKS; ++i) {
    // Use try_post for non-blocking behaviour.
    while (!pool.try_post([&completed]() {
      completed.fetch_add(1, std::memory_order_relaxed);
    })) {
      // Queue full, yield and retry.
      std::this_thread::yield();
    }
  }
  pool.wait();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double throughput = NUM_TASKS / (duration.count() / 1000.0);
  std::cout << "Completed: " << completed << " tasks\n";
  std::cout << "Time: " << duration.count() << " ms\n";
  std::cout << "Throughput: " << throughput / 1000.0 << "K tasks/sec\n\n";
}

void backpressure_example() {
  std::cout << "=== Queue Backpressure ===" << std::endl;

  // Small queue to demonstrate backpressure.
  thread_pool pool(2, 10);
  std::atomic<int> tasks_submitted{0};
  std::atomic<int> tasks_completed{0};

  for (int i = 0; i < 100; ++i) {
    if (pool.try_post([&tasks_completed, i]() {
          std::this_thread::sleep_for(10ms);
          tasks_completed.fetch_add(1, std::memory_order_relaxed);
        })) {
      tasks_submitted.fetch_add(1, std::memory_order_relaxed);
    } else {
      std::cout << "Queue full at task " << i << ", backing off...\n";
      std::this_thread::sleep_for(50ms);
      --i;
    }
  }

  pool.wait();

  std::cout << "Submitted: " << tasks_submitted << " tasks\n";
  std::cout << "Completed: " << tasks_completed << " tasks\n\n";
}

void submit_variants_example() {
  std::cout << "=== Submit Variants ===" << std::endl;
  thread_pool pool(4, 5);
  // Non-blocking try_submit.
  std::cout << "Attempting non-blocking submissions...\n";
  for (int i = 0; i < 10; ++i) {
    auto future = pool.try_submit([]() -> int {
      std::this_thread::sleep_for(100ms);
      return 42;
    });
    if (future.has_value())
      std::cout << "Task " << i << " submitted successfully\n";
    else
      std::cout << "Task " << i << " rejected (queue full)\n";
  }

  std::cout << "\nClearing queue...\n";
  pool.wait();

  // Blocking submit.
  std::cout << "\nNow using blocking submit...\n";
  for (int i = 0; i < 10; ++i) {
    auto future = pool.submit([]() -> int {
      std::this_thread::sleep_for(100ms);
      return 42;
    });
    std::cout << "Task " << i << " submitted (blocking if full)\n";
  }

  pool.wait();
  std::cout << "\n";
}

void queue_monitoring_example() {
  std::cout << "=== Queue Size Monitoring ===" << std::endl;
  thread_pool pool(2, 100);
  // Submit tasks in batches and monitor queue size.
  for (int batch = 0; batch < 5; ++batch) {
    std::cout << "Batch " << batch << ":\n";
    for (int i = 0; i < 20; ++i)
      pool.try_post([]() { std::this_thread::sleep_for(50ms); });
    std::cout << "  Queue size: " << pool.queue_size() << "\n";
    std::this_thread::sleep_for(200ms);
  }
  pool.wait();
  std::cout << "Final queue size: " << pool.queue_size() << "\n\n";
}

// Coroutine-based pipeline example.
task<std::vector<int>> generate_data(int count) {
  co_await resume_on(thread_pool::instance());
  std::vector<int> data;
  data.reserve(count);
  std::mt19937 gen(42);
  std::uniform_int_distribution<> dis(1, 100);
  for (int i = 0; i < count; ++i)
    data.push_back(dis(gen));
  co_return data;
}

task<int> process_item(int item) {
  co_await resume_on(thread_pool::instance());
  // Simulate processing.
  std::this_thread::sleep_for(1ms);
  co_return item * 2;
}

task<void> coroutine_pipeline_example() {
  std::cout << "=== Coroutine Pipeline ===" << std::endl;
  auto data = co_await generate_data(20);
  std::cout << "Generated " << data.size() << " items\n";
  // Process all items concurrently.
  std::vector<task<int>> tasks;
  for (int item : data)
    tasks.push_back(process_item(item));
  int sum = 0;
  for (auto &t : tasks)
    sum += co_await t;
  std::cout << "Processed sum: " << sum << "\n\n";
}

auto main() -> int {
  benchmark_throughput();
  backpressure_example();
  submit_variants_example();
  queue_monitoring_example();
  {
    auto coro = coroutine_pipeline_example();
    coro.resume();
    while (!coro.done()) {
      std::this_thread::sleep_for(10ms);
    }
  }
  return 0;
}
