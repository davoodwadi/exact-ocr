# Alan Wake 2 Manuscript Style Guidelines

When recreating or generating a manuscript page in the visual style of Alan Wake 2, follow these core HTML and CSS rules to ensure the aesthetic closely mimics the physical pages seen in the game, focusing on an authentic screenwriter/typewriter aesthetic.

## 1. Typography and Base Layout
*   **Font:** Always prioritize a true, clean typewriter font such as **"American Typewriter"**, **"Courier Prime"**, or standard **"Courier New"**. Avoid overtly distressed fonts like "Special Elite" unless specifically requested, as AW2 manuscripts typically start clean but are altered by formatting.
*   **Color Palette:**
    *   **Background Desk/Lighting:** Use a very dark background (e.g., `#050505`) with a harsh radial gradient (spotlight effect) illuminating the page.
    *   **Paper:** Use an off-white parchment radial gradient (e.g., `#fdfcf9` in the center fading out to darker grays like `#c4c1b6`) to mimic a bright flashlight illuminating a page in the Dark Place.
    *   **Ink (Text):** Use a stark off-black (`#111111`) to mimic heavy typewriter ribbon ink.

## 2. The Physical Paper Effect
To make the HTML element look like an 8.5x11 inch physical sheet of paper:
*   Set dimensions that match the standard paper aspect ratio (e.g., `width: 850px; min-height: 1100px;`) and use generous padding (`6rem 8rem`).
*   Apply heavy a `box-shadow` for depth, including a deep drop shadow `0 30px 60px rgba(0,0,0,0.9)` and a subtle `inset` shadow `inset 0 0 100px rgba(0,0,0,0.4)` to act as a harsh vintage vignette.
*   Add a subtle rotation (e.g., `transform: rotate(-1.2deg);`) so it doesn't look perfectly aligned with the screen.
*   **Paper Texture (Noise):** Use an embedded, visually hidden SVG filter (`feTurbulence`) to generate noise. Apply this noise via a `noise-overlay` absolute div over the paper container with `opacity: 0.25;` and `mix-blend-mode: multiply;`.

## 3. Dynamic Wake Text Effects
The distinctive style of the narrative requires specific inline elements to alter words and phrases. Use `<span>` tags with the following classes to mimic Alan's frantic typing and editing:

### `.emphasis` (Heavy Bolding)
The bold text should look like the typewriter keys were hammered forcefully into the paper.
*   `font-weight: 700;`
*   `-webkit-text-stroke: 0.8px #000;` (to artificially thicken the ink dramatically)
*   `text-shadow: 0 0 2px rgba(0,0,0,0.4);`
*   `letter-spacing: 0.01em;`

### `.scratched` (Furious Marker Strikethrough)
For words that the author scribbled out. It should look like multiple thick, misaligned dark marker strokes.
*   Use `::before` and `::after` pseudo-elements over the text block.
*   Position absolute lines vertically centered (`top: 40%` and `top: 55%`) with thick height (`3px` or `4px`).
*   Apply a slight contrasting rotation (`transform: rotate(2.5deg);` and `transform: rotate(-1.5deg);`) to make it look handwritten, thick, and erratic.

### `.smudged` (Wet/Smeared Ink)
For text that implies a hasty smudge or error.
*   Make the actual text transparent: `color: transparent;`
*   Use CSS text-shadows to build up a blurred form: `text-shadow: 0 0 5px rgba(10,10,10,0.9), 1px 1px 3px rgba(10,10,10,0.6);`

### `.misaligned` (Crooked Typewriter Keys)
For that authentic manual typewriter imperfection, occasionally drop a key or line out of alignment.
*   `position: relative;`
*   `top: 2px;`
*   `transform: rotate(1deg);`
*   `display: inline-block;`

## 4. Paragraph Formatting
*   Include standard true manuscript indentation: `text-indent: 3rem;` on paragraphs, but remove it from the first paragraph or for dramatic centered statements (`text-indent: 0;`).
*   Line height should be generous and precise (`line-height: 2;`) to represent standard double-spaced manuscripts.