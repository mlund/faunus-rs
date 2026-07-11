window.MathJax = {
  // Load boldsymbol up front rather than leaving it to autoload's lazy fetch,
  // which leaves \boldsymbol undefined and printed in red.
  loader: { load: ["[tex]/boldsymbol"] },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
  // Leave typesetting to document$ below, which also fires on each
  // instant-navigation page swap. Typesetting here as well would run a second
  // pass over the rendered output and nest every container inside itself.
  startup: { typeset: false },
};

document$.subscribe(() => {
  // Loading an extension makes startup asynchronous, so wait for it.
  MathJax.startup.promise.then(() => MathJax.typesetPromise());
});
