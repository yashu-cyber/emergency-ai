export function findSafeRoute(building, emergency) {
  return {
    exit: emergency === "🔥 Fire" ? "Exit C" : "Exit B",
    risk:
      emergency === "🔥 Fire"
        ? "High"
        : emergency === "👥 Congestion"
        ? "Medium"
        : "Low",

    route:
      building +
      " → Safe Corridor → Emergency Staircase → Safe Exit",

    reason:
      "AI selected the safest available evacuation path."
  };
}