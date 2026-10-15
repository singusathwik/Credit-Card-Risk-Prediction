export default function StatsPanel({ info, evaluation }) {
  const verified = evaluation?.metadata.model_version === info.model_version;
  return (
    <div className="stats">
      <div>
        <span>Held-out accuracy</span>
        <strong>
          {verified
            ? `${(evaluation.test.accuracy * 100).toFixed(2)}%`
            : "Not available"}
        </strong>
      </div>
      <div>
        <span>Actual defaults caught</span>
        <strong>
          {verified
            ? `${(evaluation.test.recall * 100).toFixed(2)}%`
            : "Not available"}
        </strong>
      </div>
      <div>
        <span>Held-out test records</span>
        <strong>
          {verified
            ? evaluation.metadata.split_sizes.test.toLocaleString()
            : "Not available"}
        </strong>
      </div>
      <div>
        <span>Selected model</span>
        <strong>{info.algorithm.replaceAll("_", " ")}</strong>
      </div>
    </div>
  );
}
