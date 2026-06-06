// Preload shim: make global fetch accept local filesystem paths.
//
// The mtgdraftbots WASM worker loads its `.wasm` via fetch(path) when the
// Emscripten runtime misdetects the environment. Node's undici fetch rejects
// bare filesystem paths ("Invalid URL"), so the worker fails to start. Loaded
// via NODE_OPTIONS=--require so it also runs inside worker threads.
const { readFile } = require('node:fs/promises');

const orig = globalThis.fetch;
globalThis.fetch = async (resource, options) => {
  if (typeof resource === 'string' && !/^[a-z][a-z0-9+.\-]*:\/\//i.test(resource)) {
    const data = await readFile(resource);
    const headers = resource.endsWith('.wasm') ? { 'content-type': 'application/wasm' } : {};
    return new Response(data, { headers });
  }
  return orig(resource, options);
};
