// Verify that the built documentation renders its math.
//
// Usage: node check-math.mjs <site-dir>       (e.g. after `mkdocs build -d site`)
//
// Checking the generated HTML is not enough: the markdown pipeline emits
// `\(...\)` correctly even when MathJax cannot typeset it. An undefined macro
// such as `\boldsymbol` only turns red once MathJax runs, so the pages have to
// be loaded in a real browser.
//
// Fails on:
//   - macros rendered in red   (MathJax's `noundefined` package)
//   - <mjx-merror> nodes       (TeX syntax errors)
//   - uncaught page errors     (a broken MathJax config)
//   - nested containers        (the page typeset twice)
//   - math left untypeset      (MathJax never ran)

import { createServer } from "node:http";
import { readFile, readdir } from "node:fs/promises";
import { join, extname, relative } from "node:path";
import puppeteer from "puppeteer";

const MIME = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "text/javascript",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".woff2": "font/woff2",
};

/** Serve `root` statically, resolving `/foo/` to `foo/index.html`. */
async function serve(root) {
  const server = createServer(async (req, res) => {
    let path = decodeURIComponent(req.url.split("?")[0]);
    if (path.endsWith("/")) path += "index.html";
    try {
      const body = await readFile(join(root, path));
      res.writeHead(200, { "content-type": MIME[extname(path)] ?? "application/octet-stream" });
      res.end(body);
    } catch {
      res.writeHead(404).end("not found");
    }
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  return { server, port: server.address().port };
}

/** Every page in the built site, as URL paths. */
async function pages(root) {
  const found = [];
  for (const entry of await readdir(root, { recursive: true, withFileTypes: true })) {
    if (entry.name !== "index.html") continue;
    const dir = relative(root, join(entry.parentPath ?? entry.path, ""));
    found.push("/" + (dir ? dir + "/" : ""));
  }
  return found.sort();
}

/** Load one page, let MathJax finish, and report anything wrong with the math. */
async function inspect(browser, url) {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));

  await page.goto(url, { waitUntil: "load", timeout: 60_000 });

  // Pages with no math have nothing to wait for; otherwise wait until every
  // arithmatex span holds a rendered container.
  await page.waitForFunction(
    () => {
      const spans = document.querySelectorAll(".arithmatex").length;
      return spans === 0 || document.querySelectorAll(".arithmatex mjx-container").length >= spans;
    },
    { timeout: 60_000 },
  );

  const result = await page.evaluate(() => {
    const red = [...document.querySelectorAll("mjx-container *")].filter((node) => {
      const color = getComputedStyle(node).color;
      return color === "rgb(255, 0, 0)" || color === "red";
    });
    return {
      math: document.querySelectorAll(".arithmatex").length,
      containers: document.querySelectorAll("mjx-container").length,
      nested: document.querySelectorAll("mjx-container mjx-container").length,
      merror: document.querySelectorAll("mjx-merror").length,
      red: [...new Set(red.map((n) => n.textContent.trim()).filter(Boolean))],
    };
  });
  await page.close();
  return { ...result, pageErrors: [...new Set(pageErrors)] };
}

const root = process.argv[2];
if (!root) {
  console.error("usage: node check-math.mjs <site-dir>");
  process.exit(2);
}

const { server, port } = await serve(root);
const browser = await puppeteer.launch({ args: ["--no-sandbox"] });
let failed = 0;

for (const path of await pages(root)) {
  const r = await inspect(browser, `http://127.0.0.1:${port}${path}`);
  const problems = [];
  if (r.red.length) problems.push(`undefined macro rendered in red: ${r.red.join(", ")}`);
  if (r.merror) problems.push(`${r.merror} TeX error(s)`);
  if (r.nested) problems.push(`${r.nested} nested container(s) — the page typeset twice`);
  if (r.math && !r.containers) problems.push(`${r.math} expression(s) never typeset`);
  if (r.pageErrors.length) problems.push(`page error: ${r.pageErrors.join(" | ")}`);

  const label = `${path.padEnd(24)} ${String(r.math).padStart(4)} expr`;
  if (problems.length) {
    failed += 1;
    console.log(`FAIL  ${label}`);
    for (const p of problems) console.log(`        ${p}`);
  } else {
    console.log(`ok    ${label}`);
  }
}

await browser.close();
server.close();

if (failed) {
  console.error(`\n${failed} page(s) with broken math`);
  process.exit(1);
}
console.log("\nall pages render their math");
