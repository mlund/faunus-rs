# Documentation math check

Loads the built documentation in a headless browser and fails if any page's math
is broken.

```sh
cd faunus
mkdocs build --strict -d site
npm install --prefix scripts/docs-check
node scripts/docs-check/check-math.mjs site
```

Inspecting the generated HTML is not sufficient. The markdown pipeline emits
`\(...\)` correctly even for math MathJax cannot typeset, so an undefined macro
such as `\boldsymbol` looks fine in the HTML and only turns red in the browser.

The check reports, per page:

| Failure | Meaning |
|---|---|
| macro rendered in red | undefined macro; the TeX extension is not loaded |
| `<mjx-merror>` | TeX syntax error |
| nested containers | the page was typeset twice |
| expressions never typeset | MathJax did not run |
| uncaught page error | broken MathJax configuration |

It needs network access: the pages fetch MathJax from a CDN.

Not wired into CI. To gate deploys, run it between `mkdocs build` and
`actions/upload-pages-artifact` in `.github/workflows/docs.yml`, and make the
deploy job depend on it. On GitHub runners, pass `channel: "chrome"` to
`puppeteer.launch` to use the preinstalled browser instead of downloading one.
