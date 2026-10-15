import { useEffect, useState } from "react";
import Header from "./components/Header";
import RiskForm from "./components/RiskForm";
import RiskResults from "./components/RiskResults";
import StatsPanel from "./components/StatsPanel";
import Evaluation from "./components/Evaluation";
async function post(path, payload) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const d = await r.json();
  if (!r.ok) throw Error(d.error || "Request failed");
  return d;
}
export default function App() {
  const [view, setView] = useState("predict"),
    [info, setInfo] = useState(null),
    [evaluation, setEvaluation] = useState(null),
    [prediction, setPrediction] = useState(null),
    [input, setInput] = useState(null),
    [loading, setLoading] = useState(false),
    [error, setError] = useState(""),
    [explanation, setExplanation] = useState(null),
    [explaining, setExplaining] = useState(false),
    [explainError, setExplainError] = useState("");
  useEffect(() => {
    fetch("/api/evaluation")
      .then(async (r) => {
        if (r.ok) setEvaluation(await r.json());
      })
      .catch(() => {});
    fetch("/api/model-info")
      .then(async (r) => {
        const d = await r.json();
        if (!r.ok) throw Error(d.error);
        setInfo(d);
      })
      .catch((e) => setError(e.message));
  }, []);
  async function predict(data) {
    setLoading(true);
    setError("");
    setPrediction(null);
    setExplanation(null);
    setExplainError("");
    setInput(data);
    try {
      setPrediction(await post("/api/predict", data));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }
  async function explain() {
    setExplaining(true);
    setExplainError("");
    try {
      const d = await post("/api/explain", input);
      if (d.model_version !== prediction.model_version)
        throw Error("Model changed; request a new prediction.");
      setExplanation(d.explanation);
    } catch (e) {
      setExplainError(e.message);
    } finally {
      setExplaining(false);
    }
  }
  return (
    <>
      <Header view={view} setView={setView} />
      <main>
        <section className="hero">
          <span className="eyebrow">EXPLAINABLE AI · CREDIT DEFAULT</span>
          <h1>
            Understand the risk.
            <br />
            <span>Inspect the evidence.</span>
          </h1>
          <p>
            A research demo combining historical repayment data, validated model
            comparisons and transparent probability explanations.
          </p>
          <div className="hero-tags">
            <span>Group-aware splits</span>
            <span>Calibration study</span>
            <span>SHAP explanations</span>
          </div>
        </section>
        {info && <StatsPanel info={info} evaluation={evaluation} />}
        {view === "predict" ? (
          <div className="workspace">
            <RiskForm
              initialData={input}
              onSubmit={predict}
              loading={loading || explaining}
              error={error}
            />
            <RiskResults
              prediction={prediction}
              loading={loading}
              error={error}
              explanation={explanation}
              onExplain={explain}
              explaining={explaining}
              explainError={explainError}
            />
          </div>
        ) : (
          <Evaluation />
        )}
        <footer>
          Historical Taiwan cohort · 2005 · Educational use only
          <br />
          Predictions describe learned associations, not causes or individual
          guarantees.
        </footer>
      </main>
    </>
  );
}
