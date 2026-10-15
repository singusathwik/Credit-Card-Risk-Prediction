const form = document.getElementById("prediction-form");
const submitBtn = document.getElementById("submit-btn");
const resultCard = document.getElementById("result-card");
const predictionLabel = document.getElementById("prediction-label");
const predictionProb = document.getElementById("prediction-prob");
const probWrapper = document.getElementById("prob-wrapper");
const probFill = document.getElementById("prob-fill");
const riskBadge = document.getElementById("risk-badge");
const spinner = document.getElementById("btn-spinner");

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  spinner.classList.toggle("btn-spinner--visible", isLoading);
}

function setRiskBadge(prob) {
  const pct = prob * 100;
  let label = "MEDIUM RISK";
  let cls = "risk-badge--medium";

  if (pct < 30) {
    label = "LOW RISK";
    cls = "risk-badge--low";
  } else if (pct > 70) {
    label = "HIGH RISK";
    cls = "risk-badge--high";
  }

  riskBadge.textContent = label;
  riskBadge.classList.remove("risk-badge--low", "risk-badge--medium", "risk-badge--high");
  riskBadge.classList.add(cls);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = new FormData(form);
  const payload = {};
  formData.forEach((value, key) => {
    const num = Number(value);
    payload[key] = Number.isNaN(num) || value === "" ? value : num;
  });

  setLoading(true);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    const prob = typeof data.default_probability === "number" ? data.default_probability : null;
    const willDefault = Boolean(data.will_default);

    resultCard.hidden = false;
    predictionLabel.textContent = willDefault
      ? "High risk: this customer is likely to default on their next payment cycle."
      : "Low risk: this customer is unlikely to default on their next payment cycle.";

    predictionLabel.classList.toggle("prediction-label--high", willDefault);
    predictionLabel.classList.toggle("prediction-label--low", !willDefault);

    if (prob !== null) {
      const pctText = `${(prob * 100).toFixed(1)}%`;
      predictionProb.textContent = `Estimated probability of default: ${pctText}`;
      probWrapper.hidden = false;
      probFill.style.width = `${Math.max(0, Math.min(1, prob)) * 100}%`;
      setRiskBadge(prob);
    } else {
      probWrapper.hidden = true;
      predictionProb.textContent = "";
    }
  } catch (err) {
    resultCard.hidden = false;
    predictionLabel.textContent = "Unable to get prediction from server.";
    predictionLabel.classList.remove("prediction-label--high", "prediction-label--low");
    predictionProb.textContent = err.message;
    probWrapper.hidden = true;
  } finally {
    setLoading(false);
  }
});


