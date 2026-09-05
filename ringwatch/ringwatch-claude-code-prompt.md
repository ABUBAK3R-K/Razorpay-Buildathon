# Prompt for Claude Code — RingWatch UI/UX Redesign + Landing Page

Copy everything below into Claude Code as one instruction.

---

I'm building **RingWatch**, a graph-native fraud/AML detection engine for the Razorpay AI Buildathon (Track 02: AI Risk Manager). The repo is `Razorpay-Buildathon/ringwatch/`. I need two things done well: (1) a redesign of the existing operational dashboard, and (2) a new landing page that explains the project. Read this whole brief before touching code.

## What RingWatch actually does

Most fraud engines score accounts and transactions in isolation and miss coordinated rings (mule networks, fan-out/fan-in layering, circular collusion) because each account individually stays under the radar — the signal is in the *topology*, not any single node.

RingWatch models the transaction network as a directed multigraph and runs three stages:
1. **Relational scoring** — an edge-featured GNN (GINE) flags suspicious accounts/transactions.
2. **Ring extraction** — Louvain community detection isolates candidate rings on the high-risk subgraph and classifies their shape (Fan-Out, Fan-In, Cycle, Gather-Scatter).
3. **Honest baseline benchmarking** — compared against an XGBoost tabular model on a time-based split, scored on PR-AUC and $ cost of false positives vs false negatives.

It is strictly **advisory / read-only** — it flags and explains, it never freezes accounts or blocks transactions, and every flag is written to an append-only audit trail (`GET /rings/{id}/explain` shows the reasoning).

Key numbers to actually use as real content (not filler):
- GNN recall 0.9804 vs baseline 0.9086; baseline PR-AUC 0.6113 vs GNN 0.4052
- Ring recovery: 40/52 ground-truth rings (76.9%) found via community detection
- Net FP-cost: $157,315 (baseline) vs $237,670 (GNN), under a $25/review vs $500/missed-fraud model
- The actual finding worth foregrounding: the tabular model is more cost-efficient at scoring individual transactions, while the GNN is what surfaces the *ring-level* structure the tabular model can't see. Neither model wins outright — that honesty is part of the pitch.

Current stack: FastAPI backend (`POST /score/transaction`, `GET /rings`, `GET /rings/{id}/explain`), audit storage in PostgreSQL/SQLite/JSONL, and a dashboard at `ringwatch/dashboard/index.html` built with vis-network, currently styled as a plain beige/minimal theme.

## The two deliverables

### 1. Redesign `ringwatch/dashboard/index.html` (the operational tool)

This is the working surface a risk analyst (or a buildathon judge watching a live demo) uses to look at flagged transactions, open a ring, see its topology type, and read the explanation. Keep it wired to the existing API — don't change endpoint contracts, just the front end calling them.

Right now it reads as a generic, unstyled demo. Rebuild it to feel like a real fraud-ops console: dense but legible, calm under pressure, built for someone scanning many rings quickly. The graph view (vis-network) is the centerpiece — the redesign should make the network *itself* the hero, not bury it under chrome. Concretely it needs, at minimum:
- A ring list/queue with risk score, topology type, and status, that a real analyst could triage from
- The network graph as the focal element, with clear visual encoding for edge weight, direction, and flagged nodes
- An explanation panel (features/edges contributing to the flag) that reads as an audit artifact, not a tooltip
- A visible, honest note that this is advisory-only — no action buttons that imply blocking or freezing

### 2. Build a new landing page (the pitch)

This is a separate page — not the dashboard — whose only job is to explain what RingWatch is and why it matters, to two audiences at once: a judge skimming in 30 seconds, and a technical reviewer who wants the architecture and the numbers. Put it at the repo root or in a `docs/` folder (so it can serve from GitHub Pages) and link out to the live dashboard and the GitHub repo.

Content it needs to carry (write real copy, don't leave placeholder lorem ipsum):
- A hero moment that shows, not tells — this product's actual subject matter is a graph, so the most honest hero is some form of live or illustrative network visualization (nodes/edges, a ring lighting up), not a generic headline-plus-gradient
- The problem: why per-transaction scoring misses rings
- How it works: the three-stage pipeline (scoring → extraction → benchmarking), in plain language first, technical detail available on scroll/expand
- The honest results table (numbers above) — presented as evidence, not a marketing stat
- The guardrails section — advisory-only, audit trail — this is a trust signal for a fintech risk product, treat it as such
- Links to the GitHub repo, API docs, and the dashboard itself

## Design direction — read this carefully, it matters more than usual

I don't want this to look AI-generated or templated. Before writing any code, work in two passes:

**Pass 1 — plan.** Write out a short design plan: a 4–6 color palette (named hex values), the two typefaces and their roles, a layout concept with a one-line description per section, and the design principles specific to this brief. Ground every choice in the actual subject matter — this is a fraud-graph / fintech-risk product, not a generic SaaS tool, so let that shape the palette and type choices, not a default "clean dashboard" look.

**Pass 2 — check your own plan against these generic-AI-design tells, and change anything that matches:**
- Warm cream background with a high-contrast serif and a terracotta/clay accent
- Near-black background with one bright acid-green or vermilion accent
- The "SaaS card kit" — everything chopped into identical rounded cards with the same soft grey shadow and a gradient wash behind the hero
- Tracked-out ALL-CAPS eyebrow labels above every heading, meta text joined with middle-dots, labels like "WORD — fragment," a monospace font for every small data label, arrows appended to every button/link
- Bolding or coloring a single word in a headline for "emphasis"
- Fade-and-slide-up entrance animation on every section and a hover effect on every card — pick one deliberate moment for motion (e.g., a ring assembling itself on load) instead

Only then write the code. Two typefaces max, a clear type scale, left- or asymmetric-aligned layout (a centered-everything landing page is itself a tell). Build to a quality floor without calling it out: responsive to mobile, visible keyboard focus states, respects `prefers-reduced-motion`, real color contrast.

Keep the dashboard and the landing page visually related (shared palette/type) but functionally and tonally distinct — one is a pitch, one is a tool.

## Process

1. Show me the design plan (palette, type, layout, principles) for both the dashboard redesign and the landing page before you write code.
2. Flag anything in that plan that's a generic default per the list above, and tell me what you changed.
3. Build the dashboard redesign first, then the landing page.
4. Take a screenshot of each and self-critique before calling it done — tell me what you'd cut.
