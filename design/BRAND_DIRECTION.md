# MatchPlant visual direction — Fieldwork Precision

## What this direction should communicate

MatchPlant connects field imagery, computer vision, and geospatial outputs. The visual identity should feel like a capable research instrument: precise enough for scientists, approachable enough for a first-time user. Use the aerial view of crop rows and the annotation box as recurring motifs. The product name stays **MatchPlant**; “Dashboard” describes the local app, not the brand.

The working line is **“From drone imagery to plant-level insight.”** It describes the pipeline without promising results beyond the repo's capabilities. A supporting line can explain the method: “Prepare UAV images, detect individual plants, and project results onto an orthomosaic.”

## Existing product audit

- The public [landing page](https://matchplant-dashboard.github.io/) is a small setup and local-dashboard launcher. It uses a white/gray/green palette and devotes most of the page to installation instructions. The landing page source is not in this checkout.
- The local Flask dashboard already organizes the tools by stage and provides a familiar working layout. The team prefers to retain this design.
- `dashboard/static/logo.png` presents the software ecosystem alongside the MatchPlant wordmark. Because it contains many details, display it at a size where those details remain visible.
- The repo's `modules.py` lists **11 module entries** across **4 stages** (the two step-6 paths are alternatives). Some current copy says “10 modules”; the future site should use “11 tools” or avoid a count.
- The existing workflow diagram is useful as technical documentation but too dense for a primary landing-page visual. Preserve it as a linked reference; draw a simpler four-stage illustration for the site.

## Identity system

| Role | Color | Use |
| --- | --- | --- |
| Canopy ink | `#15352D` | Headings, navigation, primary action |
| Field green | `#2F7655` | Links, progress, selected states |
| Seed highlight | `#C8E17C` | Sparse emphasis, detection markers |
| Aerial blue | `#397C91` | Geospatial layer, secondary data cues |
| Clay | `#B96F42` | Preparation-stage accent |
| Paper | `#F7F7F1` | Main page background |
| Surface | `#FFFFFF` | Cards and forms |
| Line | `#D7E1D8` | Dividers and boundaries |
| Body ink | `#253B33` | Primary text |
| Muted ink | `#566B61` | Secondary text |

Green is the brand color. Use clay, blue, and seed as wayfinding cues, not four competing brand colors. Status colors must carry labels and icons as well as color. Keep body text on paper or white. Reserve the deepest green for action buttons and strong headings.

**Typography:** a confident humanist sans for headings and interface text, with a monospace face for step numbers, data values, file names, and log output. The prototype uses local system fonts; production can use a self-hosted open font once selected. Use generous line height and restrained letter spacing. Avoid decorative script and all-caps body text.

**Shape and imagery:** subtle 12–16px card radii, thin rules, topographic/grid lines, orthomosaic tiles, small detection rectangles, and crop-row patterns. Use real aerial imagery with legible annotation overlays when suitable licensed project images are available. Avoid stock “green technology” imagery and dependency logos as hero art.

**Logo:** retain the existing MatchPlant project logo. Its detailed software marks read best at a generous size, so the shorter landing page preview places it with the research and project information and uses a text wordmark in the narrow navigation. The crop-row mark in the earlier specimen was an exploration, not a proposed replacement.

## Application scope

### Public landing page

1. Open with the work and a clear **Install on Mac or Windows** action.
2. Summarize the four-stage pipeline in one compact row.
3. Give installation its own prominent section with separate Mac and Windows steps. Explain that the dashboard and GUI tools run locally.
4. Close with the original project logo, paper citation, dataset, and source links.

### Local dashboard

Keep the current dashboard layout and workflow. Any later changes should address specific usability issues the team identifies, while preserving its familiar structure and project logo. The new landing page should help people reach and install that existing dashboard.

## Reviewable specimen

The landing page is published from the [matchplant-dashboard.github.io](https://github.com/matchplant-dashboard/matchplant-dashboard.github.io) repository. Logo variants (avatar, social preview, wordmark) are in [media/brand/](media/brand/).
