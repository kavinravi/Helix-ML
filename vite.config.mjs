export default {
  // ponytail: Rollup 4.64 loops while tracing React call arguments. Ship the small
  // amount of unused code; re-enable tree shaking after a verified upstream fix.
  build: { rollupOptions: { treeshake: false } },
};
