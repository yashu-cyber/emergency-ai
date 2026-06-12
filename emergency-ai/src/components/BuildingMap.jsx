import { buildingMaps } from "../data/buildings";

function BuildingMap({ building }) {
  const data = buildingMaps[building];

  if (!data) {
    return <div>No building data found.</div>;
  }

  return (
    <div>
      <h2>{building} Map</h2>

      <h3>Safe Route</h3>
      {data.route.map((room, index) => (
        <div key={index}>{room}</div>
      ))}

      <h3>Danger Zones</h3>
      {data.danger.map((room, index) => (
        <div key={index}>{room}</div>
      ))}

      <h3>Congestion Zones</h3>
      {data.warning.map((room, index) => (
        <div key={index}>{room}</div>
      ))}
    </div>
  );
}

export default BuildingMap;