# Triage console

The operational surface: a risk analyst opens a flagged ring, reads its topology, and works the
audit record. Served by the API at `http://127.0.0.1:8000/dashboard/`.

It is a **read-only view over the API** — it holds no data of its own and makes no call that
mutates anything. Every number on screen comes from `/metrics`, `/rings`, `/rings/{id}/graph`,
or `/rings/{id}/explain`.

## Layout

Three zones separated by hairlines. No cards, no shadows.

| Zone | Contents |
|---|---|
| Left rail | Ring queue — risk bar, topology glyph, accounts / transactions / amount, and whether the ring matched a ground-truth pattern group. Filter by id or account, by shape, and sort by risk, transactions, amount or size. |
| Centre | The transaction graph, bled edge-to-edge. Node hue encodes role, node fill encodes source vs sink, edge brightness and width encode risk, and a dashed edge is one that is *not* laundering in ground truth. |
| Right rail | `Audit record` renders `/rings/{id}/explain` as a filing — finding, provenance, account roles, contributing transactions, advisory-only clause. `Model provenance` holds the baseline-vs-GNN table, ring recovery, and extraction settings. |

## Encoding

**Risk is luminance, role is hue.** Risk score is continuous, so it rides a neutral brightness
ramp; account role is categorical, so it gets the colour. The ramp is stretched across the loaded
ring set — every ring here already cleared the 0.80 extraction threshold, so an absolute 0–1 ramp
would render the whole queue white. The absolute score is always printed beside the bar, and the
legend states the range in use.

## Interaction

- Click a queue row, or focus one and use <kbd>↑</kbd> / <kbd>↓</kbd>.
- Click a node to highlight its transactions in the audit record.
- `Fit`, `Re-layout`, and `Hide record` sit over the canvas.
- `#model` in the URL opens the provenance tab directly.

One deliberate motion moment: on selection the ring assembles itself, strongest edges first, then
holds still. Under `prefers-reduced-motion` it renders pre-stabilised.

## Guardrail

There is no control anywhere in this console that blocks, freezes, holds, or reverses anything,
because the API exposes no endpoint that could. The advisory-only contract is standing text in the
header and a clause on every audit record — see `PRD.md` §8.
