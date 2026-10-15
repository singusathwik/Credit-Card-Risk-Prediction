export default function StatsPanel({ info }) {
  return (
    <div className="stats">
      <div>
        <span>Dataset</span>
        <strong>{info.rows.toLocaleString()} clients</strong>
      </div>
      <div>
        <span>Observed defaults</span>
        <strong>{((info.positives / info.rows) * 100).toFixed(2)}%</strong>
      </div>
      <div>
        <span>Selected model</span>
        <strong>{info.algorithm.replaceAll("_", " ")}</strong>
      </div>
      <div>
        <span>Calibration</span>
        <strong>{info.calibration}</strong>
      </div>
    </div>
  );
}
