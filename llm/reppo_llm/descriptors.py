"""Terminal behaviour descriptors of a math solution (emitted once at the last token), exact text features measured on
the response up to the end of its last \\boxed{...} (the whole response if it has none): text after the final answer
is not scored. Each is tanh((x - mu) / sigma) of a raw feature x, with mu fit once on the initial policy, so the initial
population sits near 0, and sigma from descriptor_scales (shared across models), else the policy's own:
    code:         share of non-space characters inside code blocks (program-of-thought vs chain-of-thought).
    verification: prose self-verification episodes ("let me double-check", "to verify", "substituting back into
                  the original equation"; not in or about code, not search-style checking) per 1,000 words.
    symbols:      share of the non-code text (code blocks removed) inside LaTeX math; -1 = explained in words, +1 =
                  mostly notation.
    markdown:     share of the non-empty prose lines (display math and code left out) that are markdown list items,
                  headers or table rows; flowing paragraphs vs a structured, itemized layout.
"""

import re

import numpy as np

STRUCT_LINE = re.compile(r"^\s*([-*+]\s+\S|\d+[.)]\s+\S|#{1,6}\s+\S|\|.*\|)")
MATH_OPEN = {"\\[": "\\]", "$$": "$$", "\\begin{": "\\end{"}  # display-math openers and their closers
LATEX = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|\$[^$]*\$", re.S)
CODE = re.compile(r"```.*?(?:```|$)", re.S)
# Prose self-verification (verification_count); text after a prompt echo is not scored
ECHO = re.compile(r"You are a helpful assistant|Please reason step by step|\n\s*(?:User|Human|Assistant)\s*:", re.I)
CLAUSE = re.compile(r"[^\n.!?]*(?:[.!?]+|\n|$)")
VERIFY = re.compile(
    r"\b(?:let me|let's|let us|i'll|i will|i should|i want to|i can|we can|we should|we'll|we will|to|now|i'd better)"
    r"\s*,?\s*(?:(?:just|quickly|also|again|now|first|carefully|once more) )*"
    r"(?:double[- ]check|re-?check|verify|check|confirm|make sure)"
    r"|\bdouble[- ]check|\bre-?check|\bre-?verif|\bverif(?:y|ying|ication)\b|\bsanity[- ]check"
    r"|\b(?:plug|substitut)\w*\b[^.\n]{0,50}?\bback (?:in\b|into (?:the |both |each )?(?:original|given) "
    r"(?:equation|condition|system|inequalit|constraint|problem))", re.I)
CODE_REF = re.compile(r"python|\bcode\b|program|sympy|script|\boutputs?\b|\bexecut", re.I)
SOLVING = re.compile(  # search-style checking that is part of solving ("check each n")
    r"make sure (?:i|we) (?:understand|read|have|get)|\b(?:need|have) to (?:check|verify)|\bchecking \d"
    r"|\bcheck(?:s|ed|ing)?\s+(?:each|all|every|which|both|for (?:each|all)|one by one|possible|numbers?|\d"
    r"|the (?:numbers|cases|possible|conditions))", re.I)
REDO = re.compile(r"again|back|original|answer|solution|result|double|re-?check", re.I)


def markdown_share(text: str) -> float:
    """Share of the non-empty prose lines that are markdown list items, headers or table rows: display-math lines
    (\\[...\\], $$...$$, \\begin...\\end) and fenced code lines are left out of the count, so writing equations as
    bullets instead of display blocks does not raise the share."""
    n = hits = 0
    close = None  # closer of the open display block, or "```" inside a code fence
    for ln in text.split("\n"):
        t = ln.strip()
        if not t:
            continue
        if close is not None:
            if close in t:
                close = None
            continue
        if t.startswith("```"):
            close = "```"
            continue
        opener = next((o for o in MATH_OPEN if t.startswith(o)), None)
        if opener:
            close = None if MATH_OPEN[opener] in t[len(opener) :] else MATH_OPEN[opener]
            continue
        n += 1
        hits += bool(STRUCT_LINE.match(ln))
    return hits / n if n else 0.0


def share(span: re.Pattern, text: str) -> float:
    """Share of non-space characters that are inside matches of `span`."""
    inside = sum(len(re.sub(r"\s", "", m)) for m in span.findall(text))
    return inside / max(1, len(re.sub(r"\s", "", text)))


def last_boxed(text: str) -> str | None:
    """The last \\boxed{...} of `text` (braces matched), or None if there is none or it is never closed."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for i in range(start + len("\\boxed"), len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start : i + 1]
    return None


def answer_part(text: str) -> str:
    """`text` up to the end of its last \\boxed{...}, or all of it if it has none."""
    box = last_boxed(text)
    return text if box is None else text[: text.rfind(box) + len(box)]


def verification_count(text: str, gap: int = 200, near: int = 2) -> float:
    """Number of prose self-verification episodes: clauses announcing a check of one's own result ("let me
    double-check", "to verify", "substituting back into the original equation"), excluding text after a prompt echo,
    clauses in, next to (`near` clauses) or about code, and search-style checking ("check each n"); flagged clauses
    less than `gap` characters apart are one episode."""
    echo = ECHO.search(text)
    text = text[: echo.start()] if echo else text
    code = [c.span() for c in CODE.finditer(text)]
    cl = [(c.start(), c.end(), c.group(0), any(a <= c.start() < b for a, b in code))
          for c in CLAUSE.finditer(text) if c.group(0).strip()]
    n, last = 0, -gap - 1
    for i, (s, e, x, in_code) in enumerate(cl):
        near_code = any(y[3] for y in cl[max(0, i - near): i + near + 1])
        solving = SOLVING.search(x) and not REDO.search(x)
        if in_code or near_code or solving or not VERIFY.search(x) or CODE_REF.search(x):
            continue
        n += s - last > gap
        last = e
    return float(n)


def verification_rate(text: str) -> float:
    """verification_count per 1,000 words (whitespace-separated tokens) of the text before any prompt echo."""
    echo = ECHO.search(text)
    text = text[: echo.start()] if echo else text
    return 1000.0 * verification_count(text) / max(1, len(text.split()))


FEATURES = {"code": lambda t: share(CODE, t), "verification": verification_rate,
            "symbols": lambda t: share(LATEX, CODE.sub(" ", t)), "markdown": markdown_share}


class Descriptors:
    def __init__(self, names: list[str]):
        """Descriptors `names` (FEATURES keys), in this order; stats (mu, sigma per name) are set by fit or a resume."""
        self.names = list(names)
        self.stats = None

    def raw(self, texts: list[str]) -> np.ndarray:
        """(len(texts), K) raw feature values of the answer parts of `texts`, in their own units."""
        parts = [answer_part(t) for t in texts]
        return np.array([[FEATURES[n](t) for n in self.names] for t in parts]).reshape(len(texts), len(self.names))

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        """Descriptors in [-1, 1]: tanh((x - mu) / sigma) per column."""
        mu, sd = np.array([self.stats[n] for n in self.names]).reshape(-1, 2).T
        return np.tanh((raw - mu) / sd)

    def fit(self, texts: list[str], scales: dict[str, float] | None = None) -> None:
        """Freeze (mu, sigma) of each feature on the initial policy's texts (the trainer passes its EOS-terminated
        correct answers); sigma is taken from `scales` where given (fixed across models, so the behaviour spaces of
        different models share their units) and mu is always the policy's own."""
        x = self.raw(texts)
        assert (x.std(0) > 0).all(), f"a descriptor is constant on the initial policy: {self.names}"
        self.stats = {n: (float(x[:, k].mean()), float((scales or {}).get(n, x[:, k].std())))
                      for k, n in enumerate(self.names)}
