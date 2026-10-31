// Tiny pub/sub shared by every panel. See README.md for the event catalogue.
const handlers = new Map();

export const bus = {
  on(name, fn) {
    if (!handlers.has(name)) handlers.set(name, new Set());
    handlers.get(name).add(fn);
    return () => handlers.get(name).delete(fn);
  },
  off(name, fn) {
    handlers.get(name)?.delete(fn);
  },
  emit(name, data) {
    for (const fn of [...(handlers.get(name) || [])]) {
      try {
        fn(data);
      } catch (err) {
        console.error(`[bus] handler for "${name}" failed`, err);
      }
    }
  },
};
