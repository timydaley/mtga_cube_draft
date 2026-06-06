// Node bridge over the AGPL `mtgdraftbots` package (CubeArtisan/CubeCobra bot).
//
// Used ONLY offline by scripts/gen_distill_dataset.py to build a distillation
// dataset — never in the training inner loop. Speaks line-delimited JSON on
// stdin/stdout so the Python CubeCobraBot can drive it as a subprocess:
//
//   startup   -> {"ready": true}                          (after model init)
//   in  {"op":"testRecognized","cardOracleIds":[...]}      -> {"recognized":[0|1,...]}
//   in  {"op":"pick", ...drafterState}                     -> {"pick": <idx into cardOracleIds>, "score": <float>}
//   on error                                               -> {"error": "..."}
//
// `pick` is the index (into the request's cardOracleIds) of the chosen card,
// which equals the cube-local index on the Python side by construction.
//
// NOTE: the WASM worker loads its .wasm via fetch(path); modern Node rejects
// bare paths, so the caller must preload scripts/node_fetch_shim.cjs via
// NODE_OPTIONS=--require (NodeBridgeTransport does this automatically).
//
// The package is AGPL-3.0: it runs here as a separate process and we keep only
// the pick decisions (data), never vendoring its code into ours.

import readline from 'node:readline';
import { initializeDraftbots, calculateBotPick, testRecognized } from 'mtgdraftbots';

async function handle(msg) {
  if (msg.op === 'testRecognized') {
    return { recognized: await testRecognized(msg.cardOracleIds) };
  }
  if (msg.op === 'pick') {
    const { op, ...drafterState } = msg; // eslint-disable-line no-unused-vars
    const res = await calculateBotPick(drafterState);
    // res.options[res.chosenOption] is the chosen option; for calculateBotPick
    // each option is a single card given as an index into cardsInPack.
    const opt = res.options[res.chosenOption];
    const pos = Array.isArray(opt) ? opt[0] : opt;
    const score = res.scores?.[res.chosenOption]?.score ?? null;
    return { pick: drafterState.cardsInPack[pos], score };
  }
  throw new Error(`unknown op: ${msg.op}`);
}

async function main() {
  await initializeDraftbots();
  process.stdout.write(JSON.stringify({ ready: true }) + '\n');

  const rl = readline.createInterface({ input: process.stdin });
  // Serialize processing so output order matches input order.
  let chain = Promise.resolve();
  rl.on('line', (line) => {
    if (!line.trim()) return;
    chain = chain.then(async () => {
      let out;
      try {
        out = await handle(JSON.parse(line));
      } catch (e) {
        out = { error: String(e && e.message ? e.message : e) };
      }
      process.stdout.write(JSON.stringify(out) + '\n');
    });
  });
}

main().catch((e) => {
  process.stderr.write('bridge fatal: ' + e + '\n');
  process.exit(1);
});
