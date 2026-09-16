import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from readerwriterlock import rwlock


def check_readers_overlap():
    lock = rwlock.RWLockFair()
    state = {"current": 0, "max_seen": 0}
    state_lock = threading.Lock()

    def reader_task(_i):
        with lock.gen_rlock():
            with state_lock:
                state["current"] += 1
                state["max_seen"] = max(state["max_seen"], state["current"])
            time.sleep(0.2)
            with state_lock:
                state["current"] -= 1

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(reader_task, i) for i in range(5)]
        for f in as_completed(futures):
            f.result()

    print(f"max concurrent readers observed: {state['max_seen']}")
    assert state["max_seen"] > 1, (
        "readers never overlapped -- gen_rlock() is serializing reads, defeating the "
        "whole point of a reader-writer lock for concurrent population training"
    )
    print("[PASS] multiple readers genuinely overlap in time (concurrent reads confirmed)")


def check_writer_excludes_and_waits_for_readers():
    lock = rwlock.RWLockFair()
    shared_state = {"a": 0, "b": 0}  
    violations = []
    state_lock = threading.Lock()

    def reader_task():
        for _ in range(20):
            with lock.gen_rlock():
                with state_lock:
                    a, b = shared_state["a"], shared_state["b"]
                if a != b:
                    violations.append((a, b))
            time.sleep(0.005)

    def writer_task():
        time.sleep(0.05) 
        with lock.gen_wlock():
            with state_lock:
                shared_state["a"] += 1
            time.sleep(0.1)  
            with state_lock:
                shared_state["b"] += 1  

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(reader_task) for _ in range(5)]
        futures.append(pool.submit(writer_task))
        for f in as_completed(futures):
            f.result()

    print(f"invariant violations observed: {len(violations)} (expect 0)")
    assert not violations, (
        f"a reader observed partially-updated state during a write: {violations} -- "
        f"the write lock is not correctly excluding concurrent readers"
    )
    print("[PASS] no reader ever observed partially-updated state during a write "
          "(write-exclusivity confirmed)")


def check_threadpool_backpressure():
    max_workers = 3
    num_tasks = 10
    state = {"current": 0, "max_seen": 0}
    state_lock = threading.Lock()

    def task(_i):
        with state_lock:
            state["current"] += 1
            state["max_seen"] = max(state["max_seen"], state["current"])
        time.sleep(0.05)
        with state_lock:
            state["current"] -= 1

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(task, i) for i in range(num_tasks)]
        for f in as_completed(futures):
            f.result()

    print(f"max concurrent tasks observed: {state['max_seen']} (max_workers={max_workers})")
    assert state["max_seen"] <= max_workers, (
        f"observed {state['max_seen']} concurrent tasks with max_workers={max_workers} -- "
        f"the executor is not backpressuring correctly"
    )
    print(f"[PASS] ThreadPoolExecutor never exceeded max_workers={max_workers} concurrent tasks "
          f"even with {num_tasks} submitted at once")


if __name__ == "__main__":
    check_readers_overlap()
    check_writer_excludes_and_waits_for_readers()
    check_threadpool_backpressure()
    print("\nALL CONCURRENCY-PRIMITIVE CHECKS PASSED")
