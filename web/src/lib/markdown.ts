/** Safe markdown subset for chat: fenced code (with copy buttons), headings, lists, quotes, tables,
 *  rules, links, inline code, bold and italics. Everything is HTML-escaped first, so model output
 *  can't inject markup; links are limited to http(s). */
function escape(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function inline(s: string): string {
  const codes: string[] = [];
  // Code spans first, so their contents are never formatted.
  let out = escape(s).replace(/`([^`\n]+)`/g, (_, code) => `\u0000${codes.push(code) - 1}\u0000`);
  out = out
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
  return out.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
}

function codeBlock(lang: string, body: string): string {
  const label = lang ? `<span class="code-lang">${escape(lang)}</span>` : "<span></span>";
  return `<div class="code-block"><div class="code-head">${label}<button type="button" data-copy-code>Copy</button></div>` +
    `<pre><code>${escape(body.replace(/\n$/, ""))}</code></pre></div>`;
}

const TABLE_SEP = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/;

function cells(row: string): string[] {
  return row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
}

function blocks(text: string): string {
  const lines = text.split("\n");
  const out: string[] = [];
  let para: string[] = [];
  const flush = () => {
    if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`);
    para = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) { flush(); continue; }
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) { flush(); const n = Math.min(heading[1].length + 2, 6); out.push(`<h${n}>${inline(heading[2])}</h${n}>`); continue; }
    if (/^\s*(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$/.test(line)) { flush(); out.push("<hr>"); continue; }
    if (/^\s*>/.test(line)) {
      flush();
      const quote: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
      i--;
      out.push(`<blockquote>${blocks(quote.join("\n"))}</blockquote>`);
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
      flush();
      const head = cells(line);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(cells(lines[i++]));
      i--;
      out.push(`<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>` +
        rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>");
      continue;
    }
    const item = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(line);
    if (item) {
      flush();
      const ordered = /\d/.test(item[1]);
      const start = ordered ? parseInt(item[1], 10) : 1;
      const items: string[] = [];
      while (i < lines.length) {
        const m = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(lines[i]);
        if (m && /\d/.test(m[1]) === ordered) items.push(m[2]);
        else if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) items[items.length - 1] += "\n" + lines[i].trim();
        else break;
        i++;
      }
      i--;
      const tag = ordered ? "ol" : "ul";
      const open = ordered && start !== 1 ? `<ol start="${start}">` : `<${tag}>`;
      out.push(`${open}${items.map((it) => `<li>${it.split("\n").map(inline).join("<br>")}</li>`).join("")}</${tag}>`);
      continue;
    }
    para.push(line);
  }
  flush();
  return out.join("");
}

/** The same output as ``renderMarkdown``, as independent top-level blocks. While a reply streams,
 *  only its last block changes, so the earlier ones (and any code block you're scrolling or
 *  selecting in) keep their DOM. */
export function markdownBlocks(text: string): string[] {
  const out: string[] = [];
  text.split(/^```/m).forEach((part, i) => {
    if (i % 2 === 1) {
      const nl = part.indexOf("\n");
      out.push(codeBlock(nl >= 0 ? part.slice(0, nl).trim() : "", nl >= 0 ? part.slice(nl + 1) : ""));
      return;
    }
    const body = i > 0 ? part.replace(/^[^\n]*\n?/, "") : part;  // drop the rest of a closing fence line
    for (const chunk of body.split(/\n{2,}(?=\S)/)) {
      if (chunk.trim()) out.push(blocks(chunk));
    }
  });
  return out;
}

export function renderMarkdown(text: string): string {
  // An unterminated fence (still streaming) renders as code too.
  const parts = text.split(/^```/m);
  return parts
    .map((part, i) => {
      if (i % 2 === 1) {
        const nl = part.indexOf("\n");
        return codeBlock(nl >= 0 ? part.slice(0, nl).trim() : "", nl >= 0 ? part.slice(nl + 1) : "");
      }
      return blocks(part.replace(/^[^\n]*\n?/, i > 0 ? "" : "$&"));
    })
    .join("");
}
