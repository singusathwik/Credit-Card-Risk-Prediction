import { useState } from "react";
const months = ["September", "August", "July", "June", "May", "April"];
const status = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"];
const fields = [
  { name: "LIMIT_BAL", label: "Credit limit (NT$)", min: 1, max: 10000000 },
  {
    name: "SEX",
    label: "Recorded sex",
    options: [
      [1, "Male"],
      [2, "Female"],
    ],
  },
  {
    name: "EDUCATION",
    label: "Education code",
    options: [
      [1, "Graduate school"],
      [2, "University"],
      [3, "High school"],
      [4, "Other"],
      [0, "Undocumented code 0"],
      [5, "Undocumented code 5"],
      [6, "Undocumented code 6"],
    ],
  },
  {
    name: "MARRIAGE",
    label: "Marital status",
    options: [
      [1, "Married"],
      [2, "Single"],
      [3, "Other"],
      [0, "Undocumented code 0"],
    ],
  },
  { name: "AGE", label: "Age (years)", min: 18, max: 100 },
  ...status.map((name, i) => ({
    name,
    label: `${months[i]} repayment code`,
    min: -2,
    max: 9,
  })),
  ...months.map((m, i) => ({
    name: `BILL_AMT${i + 1}`,
    label: `${m} bill (NT$)`,
    min: -10000000,
    max: 10000000,
  })),
  ...months.map((m, i) => ({
    name: `PAY_AMT${i + 1}`,
    label: `${m} payment (NT$)`,
    min: 0,
    max: 10000000,
  })),
];
const empty = Object.fromEntries(fields.map((f) => [f.name, ""]));
const example = Object.fromEntries(
  fields.map((f) => [
    f.name,
    f.name === "LIMIT_BAL"
      ? 200000
      : f.name === "AGE"
        ? 35
        : f.name === "SEX"
          ? 1
          : f.name === "EDUCATION"
            ? 2
            : f.name === "MARRIAGE"
              ? 1
              : f.name.startsWith("BILL")
                ? 50000
                : f.name.startsWith("PAY_AMT")
                  ? 5000
                  : 0,
  ]),
);
export default function RiskForm({ onSubmit, loading, error, initialData }) {
  const [data, setData] = useState(initialData || empty);
  return (
    <section className="card">
      <div className="section-heading">
        <div>
          <span className="eyebrow">01 / APPLICANT PROFILE</span>
          <h2>Explore a prediction</h2>
        </div>
        <button
          type="button"
          className="secondary"
          onClick={() => setData(example)}
        >
          Load demo values
        </button>
      </div>
      <p>
        Use synthetic inputs only. Amounts are New Taiwan dollars; histories run
        September to April 2005. Recorded demographic codes are included to
        reproduce this research model.
      </p>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit(
            Object.fromEntries(
              Object.entries(data).map(([k, v]) => [k, Number(v)]),
            ),
          );
        }}
      >
        {[
          [0, 5, "Customer profile"],
          [5, 11, "Repayment history"],
          [11, 17, "Bill statements"],
          [17, 23, "Previous payments"],
        ].map(([a, b, title]) => (
          <fieldset key={title}>
            <legend>{title}</legend>
            {a === 5 && (
              <p className="hint">
                −1: paid duly; 1–9: months of delay. Observed codes −2 and 0 are
                retained as categories; their meanings are not fully defined in
                the UCI data dictionary.
              </p>
            )}
            <div className="form-grid">
              {fields.slice(a, b).map((f) => (
                <label key={f.name} htmlFor={f.name}>
                  {f.label}
                  {f.options ? (
                    <select
                      id={f.name}
                      name={f.name}
                      value={data[f.name]}
                      required
                      onChange={(e) =>
                        setData({ ...data, [f.name]: e.target.value })
                      }
                    >
                      <option value="">Select</option>
                      {f.options.map(([v, l]) => (
                        <option key={v} value={v}>
                          {l}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={f.name}
                      name={f.name}
                      type="number"
                      step="1"
                      required
                      min={f.min}
                      max={f.max}
                      value={data[f.name]}
                      onChange={(e) =>
                        setData({ ...data, [f.name]: e.target.value })
                      }
                    />
                  )}
                </label>
              ))}
            </div>
          </fieldset>
        ))}
        <p className="hint">
          Technical bounds prevent extreme inputs; they are not a claim of model
          validity outside the historical dataset (observed ages 21–79, limits
          NT$10,000–1,000,000).
        </p>
        <button className="primary" disabled={loading} type="submit">
          {loading ? "Estimating probability…" : "Estimate default probability"}
        </button>
      </form>
    </section>
  );
}
