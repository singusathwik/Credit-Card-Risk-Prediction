export default function Header({ view, setView }) {
  return (
    <header>
      <a
        className="brand"
        href="#"
        onClick={(e) => {
          e.preventDefault();
          setView("predict");
        }}
      >
        <span className="brand-mark">cr</span>
        <span>
          Credit Risk <small>RESEARCH LAB</small>
        </span>
      </a>
      <nav aria-label="Main navigation">
        <button
          aria-current={view === "predict" ? "page" : undefined}
          onClick={() => setView("predict")}
        >
          Prediction
        </button>
        <button
          aria-current={view === "evaluation" ? "page" : undefined}
          onClick={() => setView("evaluation")}
        >
          Evaluation
        </button>
      </nav>
      <span className="header-note">EXPLAINABLE ML / UCI TAIWAN</span>
    </header>
  );
}
