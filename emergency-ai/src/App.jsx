import { useState } from "react";
import "./App.css";
import BuildingMap from "./components/BuildingMap.jsx";
import { findSafeRoute } from "./utils/routing";

function App() {
  const [building, setBuilding] =
    useState("KLH University");

  const [emergency, setEmergency] =
    useState("🔥 Fire");

  const [result, setResult] =
    useState(null);

  const handleRoute = () => {
    setResult(
      findSafeRoute(
        building,
        emergency
      )
    );
  };

  return (
    <div className="app">

      <h1>SAFEPATH AI</h1>

      <div className="card">

        <select
          value={building}
          onChange={(e) =>
            setBuilding(e.target.value)
          }
        >
          <option>
            KLH University
          </option>

          <option>
            Lulu Mall
          </option>

          <option>
            Airport
          </option>
        </select>

        <select
          value={emergency}
          onChange={(e) =>
            setEmergency(e.target.value)
          }
        >
          <option>🔥 Fire</option>

          <option>
            👥 Congestion
          </option>

          <option>
            ❌ Blocked Exit
          </option>
        </select>

        <button
          onClick={handleRoute}
        >
          Find Route
        </button>
      </div>

      <BuildingMap
        building={building}
      />

      {result && (
        <div className="result">

          <h2>
            Recommended Exit:
            {" "}
            {result.exit}
          </h2>

          <p>
            Risk:
            {" "}
            {result.risk}
          </p>

          <p>
            {result.reason}
          </p>

        </div>
      )}
    </div>
  );
}

export default App;