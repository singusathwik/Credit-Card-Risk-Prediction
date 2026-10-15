import { useEffect, useState } from "react";
export default function Evaluation() {
  const [data, setData] = useState(null),
    [error, setError] = useState("");
  useEffect(() => {
    fetch("/api/evaluation")
      .then(async (r) => {
        const d = await r.json();
        if (!r.ok) throw Error(d.error);
        setData(d);
      })
      .catch((e) => setError(e.message));
  }, []);
  if (error)
    return (
      <section className="card" role="alert">
        {error}
      </section>
    );
  if (!data) return <p role="status">Loading frozen evaluation…</p>;
  const plots = [
    ["roc", "ROC curve"],
    ["precision_recall", "Precision–recall curve"],
    ["confusion_matrix", "Confusion matrix"],
    ["calibration", "Calibration"],
    ["shap_beeswarm", "Global SHAP"],
    ["thresholds", "Threshold trade-offs"],
  ];
  return (
    <section className="evaluation">
      <div className="card">
        <span className="eyebrow">FROZEN HELD-OUT EVALUATION</span>
        <h2>Evidence behind the estimate</h2>
        <p>
          On {data.metadata.split_sizes.test.toLocaleString()} held-out records,
          the model classified {(data.test.accuracy * 100).toFixed(2)}%
          correctly. It caught {Math.round(data.test.recall * 100)} out of every
          100 actual defaults and missed about{" "}
          {Math.round((1 - data.test.recall) * 100)}. Of the clients it flagged,{" "}
          {(data.test.precision * 100).toFixed(1)}% actually defaulted.
        </p>
        <p className="hint">
          Accuracy is not the chance that an individual prediction is correct.
          Catching defaults matters because most clients in this dataset did not
          default.
        </p>
        <p>
          Training / validation / test:{" "}
          {["train", "validation", "test"]
            .map((key) => data.metadata.split_sizes[key])
            .join(" / ")}
          . Identical profiles stay together. Model selection uses validation
          average precision; calibration uses validation Brier score; the
          threshold uses validation F1. Test outcomes did not choose the model.
        </p>
        <div className="metric-grid">
          {[
            "accuracy",
            "average_precision",
            "roc_auc",
            "recall",
            "precision",
            "f1",
            "brier",
          ].map((k) => (
            <div key={k}>
              <span>{k.replaceAll("_", " ")}</span>
              <strong>{data.test[k].toFixed(3)}</strong>
              <small>
                95% CI{" "}
                {data.confidence_intervals_95[k]
                  .map((v) => v.toFixed(3))
                  .join("–")}
              </small>
            </div>
          ))}
        </div>
        <p className="hint">
          300 group-bootstrap replicates; fixed-model uncertainty within this
          cohort. No prospective or geographic validation.
        </p>
      </div>
      <div className="card">
        <h2>Model comparison</h2>
        <p>
          Validation results at threshold 0.5, before calibration. CV uses only
          training folds. Ensemble membership is validation-selected, so its CV
          score is descriptive.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Candidate</th>
                <th>CV AP</th>
                <th>Validation AP</th>
                <th>Brier</th>
                <th>Fit + CV (s)</th>
              </tr>
            </thead>
            <tbody>
              {data.models.map((m) => (
                <tr key={m.algorithm}>
                  <th>{m.algorithm}</th>
                  <td>{m.cv_mean.toFixed(3)}</td>
                  <td>{m.validation.average_precision.toFixed(3)}</td>
                  <td>{m.validation.brier.toFixed(3)}</td>
                  <td>{m.training_seconds_including_cv.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="plot-grid">
        {plots.map(([name, title]) => (
          <figure className="card" key={name}>
            <figcaption>{title}</figcaption>
            <img
              src={`/api/plots/${name}.png`}
              alt={title + " for the recorded experiment"}
              loading="lazy"
            />
          </figure>
        ))}
      </div>
      <div className="card">
        <h2>Subgroup audit</h2>
        <p>
          Recorded sex codes and age bands only. Differences are descriptive;
          this does not establish fairness or authorize lending use. Small
          groups and historical sampling limit conclusions.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Attribute</th>
                <th>Group</th>
                <th>n</th>
                <th>FPR</th>
                <th>FNR</th>
                <th>Recall</th>
                <th>Brier</th>
                <th>Mean p / observed</th>
              </tr>
            </thead>
            <tbody>
              {data.subgroups.map((g) => (
                <tr key={g.attribute + g.group}>
                  <td>{g.attribute}</td>
                  <td>{g.group}</td>
                  <td>{g.n}</td>
                  <td>{g.false_positive_rate?.toFixed(3)}</td>
                  <td>{g.false_negative_rate?.toFixed(3)}</td>
                  <td>{g.recall.toFixed(3)}</td>
                  <td>{g.brier.toFixed(3)}</td>
                  <td>
                    {g.mean_probability.toFixed(3)} /{" "}
                    {g.observed_rate.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
