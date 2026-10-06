// Builds dist/ (load it with chrome://extensions -> Load unpacked).
import { build } from "esbuild";
import { cpSync, mkdirSync, readdirSync, rmSync } from "node:fs";

rmSync("dist", { recursive: true, force: true });
mkdirSync("dist/wasm", { recursive: true });

// config.js stays a separate file so you can edit it in dist/ without rebuilding
const externalConfig = { name: "external-config", setup(b) { b.onResolve({ filter: /^\.\/config\.js$/ }, () => ({ path: "./config.js", external: true })); } };

await build({ entryPoints: ["src/offscreen.js"], bundle: true, format: "esm", platform: "browser", target: "chrome116",
  outfile: "dist/offscreen.js", plugins: [externalConfig], minify: true, legalComments: "none", logLevel: "warning" });

for (const f of ["manifest.json", "offscreen.html", "background.js", "content.js", "config.js"]) cpSync(`src/${f}`, `dist/${f}`);
cpSync("icons", "dist/icons", { recursive: true });
const ort = "node_modules/onnxruntime-web/dist";
for (const f of readdirSync(ort)) if (/^ort-wasm-simd-threaded(\.jsep)?\.(wasm|mjs)$/.test(f)) cpSync(`${ort}/${f}`, `dist/wasm/${f}`);
console.log("built dist/");
