import { useEffect, useState } from "react";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const REFRESH_INTERVAL = 5000;

function StatusCard({ title, value, description, variant = "neutral" }) {
  return (
    <div className="status-card">
      <div className="card-heading">
        <span>{title}</span>
        <span className={`status-dot ${variant}`} />
      </div>

      <h2>{value}</h2>
      <p>{description}</p>
    </div>
  );
}

function App() {
  const [backend, setBackend] = useState(null);
  const [database, setDatabase] = useState(null);
  const [aiops, setAiops] = useState(null);
  const [incidents, setIncidents] = useState(null);

  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [errors, setErrors] = useState({});

  useEffect(() => {
    let active = true;

    async function fetchJSON(endpoint) {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      return response.json();
    }

    async function fetchDashboardData() {
      const endpoints = [
        ["/health", setBackend, "backend"],
        ["/health/database", setDatabase, "database"],
        ["/aiops/status", setAiops, "aiops"],
        ["/aiops/incidents", setIncidents, "incidents"],
      ];

      const results = await Promise.allSettled(
        endpoints.map(([endpoint]) => fetchJSON(endpoint))
      );

      if (!active) return;

      const nextErrors = {};

      results.forEach((result, index) => {
        const [, setter, key] = endpoints[index];

        if (result.status === "fulfilled") {
          setter(result.value);
        } else {
          setter(null);
          nextErrors[key] = result.reason?.message || "Request failed";
        }
      });

      setErrors(nextErrors);
      setLastUpdated(new Date());
      setLoading(false);
    }

    fetchDashboardData();

    const interval = setInterval(fetchDashboardData, REFRESH_INTERVAL);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const backendHealthy = backend?.status === "healthy";
  const databaseConnected = database?.status === "connected";
  const modelLoaded = aiops?.artifacts_loaded === true;

  const windowSize = aiops?.window_size ?? 0;
  const windowCapacity = aiops?.window_capacity ?? 10;

  const detectorReady = aiops?.detector_ready === true;

  const latestResult = aiops?.latest_detection_result;

  const anomalyDetected =
    detectorReady && latestResult?.is_anomaly === true;

  const reconstructionError = latestResult?.reconstruction_error;
  const threshold = aiops?.saved_threshold;

  const progress = Math.min(
    100,
    Math.round((windowSize / windowCapacity) * 100)
  );

  let anomalyLabel = "Awaiting Data";
  let anomalyVariant = "neutral";

  if (detectorReady && latestResult?.window_ready) {
    anomalyLabel = anomalyDetected ? "Anomaly Detected" : "Normal";
    anomalyVariant = anomalyDetected ? "danger" : "success";
  } else if (aiops) {
    anomalyLabel = "Warming Up";
    anomalyVariant = "warning";
  }

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-icon">◈</span>
          <div>
            <strong>Explainable AIOps</strong>
            <small>Monitoring Platform</small>
          </div>
        </div>

        <nav className="sidebar-nav" aria-label="Dashboard navigation">
          <a href="#overview" className="nav-active">
            Overview
          </a>
          <a href="#anomaly">Anomaly Detection</a>
          <a href="#incidents">Incidents</a>
        </nav>

        <div className="sidebar-footer">
          <span className="sidebar-label">Environment</span>
          <strong>Local Development</strong>
        </div>
      </aside>

      <main className="dashboard-main" id="overview">
        <header className="topbar">
          <div>
            <p className="eyebrow">SYSTEM MONITORING</p>
            <h1>Dashboard Overview</h1>
            <p className="subtitle">
              Live monitoring for your e-commerce AIOps application
            </p>
          </div>

          <div className="live-badge">
            <span className={`status-dot ${backendHealthy ? "success" : "danger"}`} />
            {backendHealthy ? "Backend Online" : "Backend Offline"}
          </div>
        </header>

        <section className="section">
          <div className="section-heading">
            <h2>System Health</h2>
            <span>Auto-refresh: 5 seconds</span>
          </div>

          <div className="cards-grid">
            <StatusCard
              title="Backend API"
              value={
                loading
                  ? "Checking..."
                  : backendHealthy
                    ? "Healthy"
                    : "Unavailable"
              }
              description="FastAPI service status"
              variant={backendHealthy ? "success" : "danger"}
            />

            <StatusCard
              title="PostgreSQL Database"
              value={
                loading
                  ? "Checking..."
                  : databaseConnected
                    ? "Connected"
                    : "Unavailable"
              }
              description="Database connectivity"
              variant={databaseConnected ? "success" : "danger"}
            />

            <StatusCard
              title="LSTM Autoencoder"
              value={
                loading
                  ? "Checking..."
                  : modelLoaded
                    ? "Loaded"
                    : "Unavailable"
              }
              description="Saved model artifacts"
              variant={modelLoaded ? "success" : "danger"}
            />

            <StatusCard
              title="Recorded Incidents"
              value={
                incidents?.incident_count ?? (loading ? "..." : "—")
              }
              description="Current in-memory incident history"
              variant="neutral"
            />
          </div>
        </section>

        <section className="section" id="anomaly">
          <div className="section-heading">
            <h2>Anomaly Detection</h2>
            <span>Live LSTM monitoring</span>
          </div>

          <div className="panel">
            <div className="anomaly-header">
              <div>
                <p className="panel-label">LATEST DETECTION RESULT</p>
                <h3>{anomalyLabel}</h3>
              </div>

              <span className={`status-pill ${anomalyVariant}`}>
                {detectorReady ? "Detector Ready" : "Detector Not Ready"}
              </span>
            </div>

            <div className="metrics-grid">
              <div className="metric">
                <span>Reconstruction Error</span>
                <strong>
                  {typeof reconstructionError === "number"
                    ? reconstructionError.toFixed(6)
                    : "Not available"}
                </strong>
              </div>

              <div className="metric">
                <span>Anomaly Threshold</span>
                <strong>
                  {typeof threshold === "number"
                    ? threshold.toFixed(6)
                    : "Not available"}
                </strong>
              </div>

              <div className="metric">
                <span>Window Progress</span>
                <strong>
                  {windowSize} / {windowCapacity}
                </strong>
              </div>
            </div>

            <div className="progress-heading">
              <span>Observation window</span>
              <span>{progress}%</span>
            </div>

            <div
              className="progress-track"
              role="progressbar"
              aria-valuenow={progress}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                className="progress-fill"
                style={{ width: `${progress}%` }}
              />
            </div>

            <p className="panel-note">
              {detectorReady
                ? "The observation window is ready for anomaly detection."
                : "The detector is collecting observations before its first prediction."}
            </p>
          </div>
        </section>

        <section className="section" id="incidents">
          <div className="section-heading">
            <h2>Incident Monitoring</h2>
            <span>Explainable anomaly history</span>
          </div>

          <div className="panel incident-panel">
            <div className="incident-summary">
              <div>
                <p className="panel-label">INCIDENTS IN MEMORY</p>
                <h3>{incidents?.incident_count ?? "—"}</h3>
              </div>

              <span className="status-pill neutral">
                Maximum capacity: {incidents?.max_capacity ?? 100}
              </span>
            </div>

            <p className="panel-note">
              {incidents?.incident_count > 0
                ? "Anomaly incidents are available for detailed inspection."
                : "No anomaly incidents are currently recorded."}
            </p>

            <p className="disclaimer">
              Feature contributions indicate reconstruction-error contributions,
              not confirmed root causes.
            </p>
          </div>
        </section>

        {Object.keys(errors).length > 0 && (
          <section className="error-banner" role="alert">
            Some monitoring endpoints could not be reached:
            {" "}
            {Object.entries(errors)
              .map(([name, message]) => `${name}: ${message}`)
              .join(" | ")}
          </section>
        )}

        <footer className="dashboard-footer">
          <span>Explainable AIOps Framework</span>
          <span>
            Last updated:{" "}
            {lastUpdated ? lastUpdated.toLocaleTimeString() : "Waiting..."}
          </span>
        </footer>
      </main>
    </div>
  );
}

export default App;