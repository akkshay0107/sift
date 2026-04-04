# Memory Engine GUI Specification: "The Quake Dropdown"

## Core Concept: "Top-Down Intelligence"
The Memory Engine GUI is a high-performance utility designed for instant access and zero-friction searching. Inspired by the classic "Quake" terminal, the interface is a fullscreen-width (or wide-centered) panel that slides down from the top edge of the primary monitor. This layout maximizes vertical scrolling space for "bundle" results while staying out of the way when inactive.

### Visual Aesthetic: "Semi-Transparent Overlay"
- **Terminal Influence**: A dark, slightly translucent charcoal background (`#121212` at 95% opacity).
- **Fixed-Width Layout**: While it slides from the top, the content is centered in a readable column (e.g., 900px wide).
- **High-Contrast Typography**: System-default sans-serif for headings and a crisp, high-resolution monospace font for metadata and technical details.
- **Minimalist Accents**: A single horizontal 1px accent line (e.g., Slate or Muted Blue) that separates the search input from the result list.

---

## Screen Architecture

### 1. The "Drop-Down" Interface
A panel that covers the top 40% to 60% of the screen when active.
- **Trigger**: Global macro (`Cmd + ~` or a custom shortcut).
- **Animation**: A high-speed, linear slide-down from the top bezel.
- **Search Header**: A single, wide input field with a blinking block cursor and no borders, resembling a command-line interface.

### 2. The "Bundle Stream"
Search results are presented as a continuous vertical stream of "Bundles."
- **Clustered Entries**: Each bundle is a group of related files (Text, Audio, Image) with a small "Group ID" or "Cluster Tag."
- **Modality Tags**: Minimalist, text-based tags like `[TXT]`, `[AUD]`, or `[IMG]` next to filenames.
- **Metadata Snippets**: A 2-line preview of OCR text or transcript snippets below the filename.

### 3. Key Interactions (Keyboard-First)
The app is designed to be used entirely without a mouse.
- `Up/Down Arrows`: Navigate through the result stream.
- `Enter`: Open the primary file in the bundle.
- `Cmd + Right`: Expand a bundle to see all secondary related files.
- `Cmd + C`: Copy the path of the highlighted file.
- `Esc`: Instantly retract the terminal to the top.

### 4. The "Status Line" (Bottom Edge)
A 20px tall bar at the very bottom of the dropdown panel.
- Shows total indexed vectors, server status, and current latency.
- Displays "Matches: 12" on the right side.

---

## Implementation Strategy
- **Framework**: **Tauri v2** (using the `tauri-plugin-positioner` to anchor the window to the top-center).
- **Styling**: Vanilla CSS with `vh` units for height control.
- **Shortcut Handling**: Global hotkey listener in the Rust backend for instant response.
