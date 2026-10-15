export default function RiskResults({
  prediction,
  loading,
  error,
  explanation,
  onExplain,
  explaining,
  explainError,
}) {
  return (
    <aside className="card results" aria-live="polite">
      <span className="eyebrow">02 / MODEL OUTPUT</span>
      <h2>Risk estimate</h2>
      {loading ? (
        <p role="status">Computing your estimate…</p>
      ) : error ? (
        <p className="error">{error}</p>
      ) : !prediction ? (
        <div className="empty">
          <div className="empty-mark">%</div>
          <h3>A probability, with context</h3>
          <p>
            Complete the profile to see an estimate and the features that
            influence it.
          </p>
        </div>
      ) : (
        <>
          <div className="probability">
            {(prediction.default_probability * 100).toFixed(1)}
            <span>%</span>
          </div>
          <p>Estimated next-month default probability</p>
          <span className="badge">
            {prediction.risk_category} risk · illustrative band
          </span>
          <div className="meter">
            <div
              style={{ width: `${prediction.default_probability * 100}%` }}
            />
          </div>
          <dl>
            <dt>Classification threshold</dt>
            <dd>{(prediction.decision_threshold * 100).toFixed(0)}%</dd>
            <dt>Threshold result</dt>
            <dd>
              {prediction.predicted_label
                ? "Above threshold"
                : "Below threshold"}
            </dd>
          </dl>
          <p className="hint">
            Bands: low &lt; {prediction.risk_boundaries[0] * 100}%, moderate
            &lt; {prediction.risk_boundaries[1] * 100}%, high otherwise. These
            are illustrative display categories, separate from the
            validation-selected classification threshold.
          </p>
          <details>
            <summary>Model version</summary>
            <code>{prediction.model_version}</code>
          </details>
          <hr />
          <h3>Why this estimate?</h3>
          <p>
            SHAP explains how inputs move this model’s probability relative to a
            training-background average.
          </p>
          <button
            className="secondary"
            onClick={onExplain}
            disabled={explaining}
          >
            {explaining ? "Computing SHAP…" : "Explain this prediction"}
          </button>
          {explainError && (
            <p role="alert" className="error">
              {explainError}
            </p>
          )}
          {explanation && (
            <div className="explanation">
              <p>
                Background probability:{" "}
                {(explanation.base_value * 100).toFixed(1)}%
              </p>
              {explanation.contributions.slice(0, 8).map((c) => (
                <div className="contribution" key={c.feature}>
                  <div>
                    <strong>{c.feature}</strong>
                    <span>
                      {c.contribution >= 0 ? "+" : ""}
                      {(c.contribution * 100).toFixed(2)} pp
                    </span>
                  </div>
                  <div className="contribution-track">
                    <i
                      className={c.contribution >= 0 ? "increase" : "decrease"}
                      style={{
                        width: `${Math.min(100, Math.abs(c.contribution) * 500)}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
              <p className="hint">
                pp = percentage points of model probability. Top 8 shown; all 23
                contributions are returned by the API. Positive increases the
                estimate; negative reduces it. Approximate, non-causal and
                sensitive to correlated features.
              </p>
            </div>
          )}
        </>
      )}
      <div className="notice">
        Educational research prototype. Estimates are not guarantees. Do not use
        as the sole basis for lending decisions.
      </div>
    </aside>
  );
}
