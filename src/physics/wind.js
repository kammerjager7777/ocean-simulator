function gustFactor(mode, time) {
  if (mode !== "gusty") return 0;
  return 0.35 * Math.sin(time * 0.7) * Math.sin(time * 0.17 + 1.3);
}

export function createWind(initial) {
  const lagged = new Map();
  return {
    direction: initial.direction,
    speed: initial.speed,
    gust: initial.gust,
    sample(band, time, dt) {
      const instant = Math.max(0, this.speed * (1 + gustFactor(this.gust, time)));
      const prev = lagged.has(band.id) ? lagged.get(band.id) : instant;
      const tau = Math.max(band.responseSeconds, 0.05);
      const follow = dt <= 0 ? 1 : 1 - Math.exp(-dt / tau);
      const speed = prev + (instant - prev) * follow;
      lagged.set(band.id, speed);
      return { direction: this.direction, speed };
    },
  };
}
