const fs = require("fs");
const path = require("path");
const vm = require("vm");

const renderPath = path.join(__dirname, "..", "frontend", "js", "render.js");
const code = fs.readFileSync(renderPath, "utf8");

let parseCalls = [];
const ctx = {
  window: { ICS: {} },
  marked: {
    parse: (source) => {
      parseCalls.push(source.length);
      return "<article>" + source.length + "</article>";
    },
  },
  DOMPurify: { sanitize: (source) => source },
  renderMathInElement: () => {},
};
ctx.window.window = ctx.window;
vm.createContext(ctx);
vm.runInContext(code, ctx);

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

const render = ctx.window.ICS.render;
const longSummary = [
  "### 复习优先级判断\n\n",
  "| 级别 | 内容 |\n|---|---|\n| 必须掌握 | 线粒体、叶绿体、ATP 合成 |\n\n",
  "A".repeat(45000),
].join("");

assert(
  typeof render.renderMarkdownPage === "function",
  "renderMarkdownPage should be exported for paged summary rendering",
);

parseCalls = [];
const firstPage = render.renderMarkdownPage(longSummary, 1, 12000);
assert(firstPage.isPaged === true, "long summary should be paged");
assert(firstPage.page === 1, "first page should stay on page 1");
assert(firstPage.pageCount === 4, "page count should be ceil(total/pageSize)");
assert(firstPage.startChar === 1, "first page should start at char 1");
assert(firstPage.endChar === 12000, "first page should end at page size");
assert(firstPage.html.includes("<article>"), "page should use markdown renderer");
assert(!firstPage.html.includes("<pre"), "paged markdown should not fall back to raw pre preview");
assert(parseCalls.length === 1, "first page should call markdown renderer once");
assert(parseCalls[0] <= 12000, "markdown renderer input should be bounded to one page");

parseCalls = [];
const lastPage = render.renderMarkdownPage(longSummary, 99, 12000);
assert(lastPage.page === 4, "page should clamp to last page");
assert(lastPage.endChar === longSummary.length, "last page should end at total chars");
assert(parseCalls.length === 1, "last page should call markdown renderer once");
assert(parseCalls[0] <= 12000, "last page renderer input should remain bounded");

console.log("frontend render paging ok");
