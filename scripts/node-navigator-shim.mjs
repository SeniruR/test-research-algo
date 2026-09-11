/**
 * Node 22+ exposes `navigator` (and sometimes `location`) as getter-only
 * globals. drawio-mcp's jsdom bootstrap still does `global.navigator = ...`,
 * which crashes. Make those properties writable before the package loads.
 */
for (const name of ["navigator", "location"]) {
  if (!(name in globalThis)) continue;
  try {
    Object.defineProperty(globalThis, name, {
      value: globalThis[name],
      writable: true,
      configurable: true,
      enumerable: true,
    });
  } catch {
    // Already locked down in a way we cannot override; the next assign will fail.
  }
}
