import { buildingMaps } from "../data/buildings";

function BuildingMap({ building }) {
  const data = buildingMaps[building];

  if (!data) return null;

  return (
    <div className="map-card">
      <h2>{building} Map</h2>

      <h3>🟢 Safe Route</h3>

      {data.route.map((room, index) => (
        <div key={index}>
          <div className="room safe">{room}</div>

          {index !== data.route.length - 1 && (
            <div className="arrow">↓</div>
          )}
        </div>
      ))}

      <h3 style={{ marginTop: "25px" }}>
        🔴 Danger Zones
      </h3>

      {data.danger.map((room, index) => (
        <div key={index} className="room danger">
          {room}
        </div>
      ))}

      <h3 style={{ marginTop: "25px" }}>
        🟡 Congestion Zones
      </h3>

      {data.warning.map((room, index) => (
        <div key={index} className="room warning">
          {room}
        </div>
      ))}
    </div>
  );
}

export default BuildingMap;