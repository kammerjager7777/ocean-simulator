// Scenario one is data: open ocean, straight fetch, uniform depth.
// Frequency bounds are fractions of the wind peak. Nothing here assumes how many bands exist.

export const scenario = {
  id: "open-ocean",
  label: "Open ocean, straight fetch",
  depth: 40,
  size: 250,
  resolution: 192,
  breakSlope: 0.2,
  depthBreakRatio: 0.78,
  choppiness: 1.5,
  crestReach: 4,
  // Seconds foam lingers after a sample was last flagged, fading smoothly to nothing.
  foamFadeSeconds: 2,
  bands: [
    {
      id: "swell",
      label: "Low-frequency swell",
      omegaMin: 0.72,
      omegaMax: 0.95,
      frequencies: 4,
      directions: 1,
      spread: 0.18,
      responseSeconds: 28,
      spectrum: "pm",
    },
    {
      id: "wind-sea",
      label: "Mid-frequency wind sea",
      omegaMin: 0.95,
      omegaMax: 1.7,
      frequencies: 4,
      directions: 3,
      spread: 0.55,
      responseSeconds: 8,
      spectrum: "pm",
    },
    {
      id: "chop",
      label: "Short wind chop",
      omegaMin: 1.7,
      omegaMax: 5.5,
      frequencies: 6,
      directions: 5,
      spread: 1.2,
      responseSeconds: 2,
      // Elfouhaily et al. 1997 over the whole band (grid components and both patches), one
      // formula, no splice. Its long-wave part takes its level from PM at the band's lower
      // edge, where it hands over from the wind-sea band (matchBelow), so there is no step
      // in energy there. "pm" switches the band back to Pierson-Moskowitz.
      spectrum: "elfouhaily",
      matchBelow: "pm",
      // The break test sees this band at its PM level, whichever spectrum is drawn: a
      // steeper chop spectrum must not spread foam across the sea. (false would leave the
      // band out of the break test entirely; at 17.5 m/s that stops all breaking, because
      // this band's grid waves, 8 to 88 m long, are the ones that tip over.)
      breakTest: "pm",
      // The chop band's own fine patch: a 56 m square at 128 samples (0.44 m apart),
      // tiled over the sea. It holds the chop wavelengths the big grid does not, from
      // just above what the grid holds down to 2.4 patch cells (about 1.05 m), on the
      // same spectrum. Used for shading; break detection stays on the big grid.
      patch: { size: 56, resolution: 128 },
      // Its second, finer patch: a 13.5 m square at 128 samples (0.105 m apart). It carries
      // the same band on down from where the first patch stops (1.05 m) to 2.4 of its own
      // cells (about 0.25 m). Shading only, like the first.
      finePatch: { size: 13.5, resolution: 128 },
    },
  ],
};
