/** Tiny, safe markdown subset for chat: fenced code, inline code, bold, italics, paragraphs, line breaks.
 *  Everything is HTML-escaped first, so model output can't inject markup. */
function escape(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function inline(s: string): string {
  return escape(s)
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
}

export function renderMarkdown(text: string): string {
  const parts = text.split(/```/);
  return parts
    .map((part, i) => {
      if (i % 2 === 1) {
        const nl = part.indexOf("\n");
        const body = nl >= 0 ? part.slice(nl + 1) : part;
        return `<pre><code>${escape(body.replace(/\n$/, ""))}</code></pre>`;
      }
      return part
        .split(/\n{2,}/)
        .filter((p) => p.trim())
        .map((p) => `<p>${inline(p).replace(/\n/g, "<br>")}</p>`)
        .join("");
    })
    .join("");
}
