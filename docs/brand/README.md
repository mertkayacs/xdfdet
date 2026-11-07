# xdfdet logo

**The idea:** the name has two identical d's, like a real frame and its deepfake; the logo flags one of them, its bowl
holding a gold core inside a contour line, a heat map reduced to one level that shows where the evidence is.

The flagged d is the d that sits second in the name (the d of "df"). The wordmark always carries the flag, and the
mark is that same d on its own.

## Files

| File | Use |
|---|---|
| `svg/xdfdet-logo-on-light.svg` | Main logo on light grounds. The flagged d sits inside the name. Height 256. |
| `svg/xdfdet-logo-on-dark.svg` | Main logo on dark grounds (#0D1117, #1F2430, navy). |
| `svg/xdfdet-logo-stacked.svg` | Mark above the name, for square spaces and covers. |
| `svg/xdfdet-mark.svg` | The mark on light grounds, from 32 px up. |
| `svg/xdfdet-mark-on-dark.svg` | The mark on dark grounds (#0D1117 and #1F2430), slightly thinner so it looks the same weight. |
| `svg/xdfdet-mark-black.svg`, `svg/xdfdet-mark-white.svg` | One-colour versions for print, stamps, engraving. |
| `svg/xdfdet-sticker.svg` | The mark with its white die-cut border, for the sticker sheet. |
| `svg/xdfdet-appicon.svg` | App icon: the mark on a navy rounded tile, 1024 canvas. |
| `favicon/favicon.svg` | Browser tab icon: a small-size cut of the d, drawn to the 16 px grid. It switches to the paper colour in dark mode. |
| `favicon/favicon.ico`, `favicon/favicon-32.png` | The small cut on a navy tile, for places that cannot follow dark mode. |
| `favicon/apple-touch-icon.png`, `favicon/icon-192.png`, `favicon/icon-512.png`, `favicon/icon-maskable-512.png` | Home-screen and web-app icons on full-bleed navy; the maskable icon keeps the mark inside the 80 % safe zone. |
| `png/` | PNG exports rendered from the SVGs. |

## Colours

Contrast is the WCAG ratio, measured from the HEX values.

| Colour | HEX | Use | on #FFFFFF | on #FDF8F1 | on #0D1117 |
|---|---|---|---|---|---|
| Navy | `#1E2A38` | Letters on light grounds, tiles | 14.55 : 1 | 13.77 : 1 | 1.30 : 1 (do not use) |
| Gold | `#A38046` | The evidence core, nothing else | 3.66 : 1 | 3.47 : 1 | 5.17 : 1 |
| Paper | `#F3ECE2` | Letters on dark grounds | 1.17 : 1 (do not use) | 1.11 : 1 (do not use) | 16.14 : 1 |

Gold is a graphic, never text, so the 3 : 1 rule for graphics applies; it passes on all three grounds. On #1F2430,
navy measures 1.07 : 1 and paper 13.23 : 1, so dark sheets take the on-dark files or the sticker.

## Clear space

Keep the height of the x in "xdfdet" clear on every side of the logo. Around the mark alone, keep the height of its
gold core clear.

## Minimum sizes

These were checked on screen:

- Logo: 96 px wide.
- Stacked logo: 120 px wide.
- Mark: 32 px.
- Small cut (`favicon.svg`): 16 px.
- Sticker: 24 px.

Below 32 px use the small cut. Print sizes have not been proofed.

## Font and licence

The wordmark is Newsreader by Production Type (variable font from the google/fonts repository; wordmark at optical
size 24 and weight 560, mark at weight 600, reversed files 20 weight units lighter, favicon cut at optical size 6). The
letters are outlined, so no font is needed to show the logo. Newsreader is licensed under the SIL Open Font License
1.1 (Copyright 2020 The Newsreader Project Authors). The licence's own condition 5 says its terms do not apply to
documents created using the font, which covers this artwork. The site already sets its headings in Newsreader.

## Do

- Use the flagged wordmark as the main logo, and the mark where the space is square or small.
- Use the on-dark files on dark grounds, and the sticker file on coloured sheets.
- Keep gold for the core alone.
- Keep the contour line between the core and the letter; it is part of the idea.
- Use `favicon.svg` for tabs and anything under 32 px.

## Don't

- Flag both d's, or move the flag to another letter.
- Recolour the core, fill the contour line, or put the navy logo on a dark ground.
- Stretch, rotate, outline, shadow, glow or add a gradient to any part.
- Retype "xdfdet" in a font, live or not; use the files.
- Use the full mark below 32 px.
