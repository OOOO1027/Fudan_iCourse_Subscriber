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
assert(firstPage.html.includes("<pre"), "paged long summaries should render as visible text chunks");
assert(firstPage.html.includes("复习优先级判断"), "first page should show summary content");
assert(parseCalls.length === 0, "paged long summaries should not call markdown renderer");

parseCalls = [];
const lastPage = render.renderMarkdownPage(longSummary, 99, 12000);
assert(lastPage.page === 4, "page should clamp to last page");
assert(lastPage.endChar === longSummary.length, "last page should end at total chars");
assert(lastPage.html.includes("<pre"), "last page should also render as visible text");
assert(parseCalls.length === 0, "last page should not call markdown renderer");

parseCalls = [];
const singleLine = "| 级别 | 内容 |" + "B".repeat(30000);
const secondPage = render.renderMarkdownPage(singleLine, 2, 12000);
assert(secondPage.page === 2, "single-line content should move to page 2");
assert(secondPage.html.includes("B"), "page 2 of a long single line should not be blank");
assert(parseCalls.length === 0, "single-line long page should not call markdown renderer");

const padded = "| 级别 | 内容 |" + " ".repeat(30000) + "核心内容";
const paddedPage = render.renderMarkdownPage(padded, 1, 12000);
assert(paddedPage.html.includes("核心内容"), "display chunks should compact excessive padding");
assert(paddedPage.rawTotalChars === padded.length, "raw total should preserve original length metadata");
assert(paddedPage.totalChars < paddedPage.rawTotalChars, "display total should reflect compacted text length");

console.log("frontend render paging ok");
