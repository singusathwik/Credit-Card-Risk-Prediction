import { AlertCircle, CheckCircle2, AlertTriangle, TrendingUp, BarChart3 } from 'lucide-react'

export default function RiskResults({ prediction, loading, error }) {
  if (loading) {
    return (
      <div className="glass rounded-2xl p-8 shadow-xl sticky top-24">
        <div className="flex flex-col items-center justify-center py-12">
          <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
          <p className="text-slate-600 font-medium">Analyzing risk profile...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="glass rounded-2xl p-8 shadow-xl sticky top-24">
        <div className="flex flex-col items-center justify-center py-8">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mb-4">
            <AlertCircle className="w-8 h-8 text-red-600" />
          </div>
          <p className="text-red-700 font-medium">{error}</p>
        </div>
      </div>
    )
  }

  if (!prediction) {
    return (
      <div className="glass rounded-2xl p-8 shadow-xl sticky top-24">
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <BarChart3 className="w-8 h-8 text-slate-400" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 mb-2">Risk Assessment</h3>
          <p className="text-sm text-slate-600">
            Fill out the form and submit to get instant risk analysis
          </p>
        </div>
      </div>
    )
  }

  const prob = prediction.default_probability || 0
  const probPercent = (prob * 100).toFixed(1)
  const willDefault = prediction.will_default || prob >= 0.5

  let riskLevel = 'LOW'
  let riskColor = 'green'
  let riskBg = 'bg-green-50'
  let riskBorder = 'border-green-200'
  let riskText = 'text-green-700'
  let Icon = CheckCircle2

  if (prob >= 0.7) {
    riskLevel = 'HIGH'
    riskColor = 'red'
    riskBg = 'bg-red-50'
    riskBorder = 'border-red-200'
    riskText = 'text-red-700'
    Icon = AlertCircle
  } else if (prob >= 0.4) {
    riskLevel = 'MEDIUM'
    riskColor = 'amber'
    riskBg = 'bg-amber-50'
    riskBorder = 'border-amber-200'
    riskText = 'text-amber-700'
    Icon = AlertTriangle
  }

  return (
    <div className="glass rounded-2xl p-8 shadow-xl sticky top-24">
      <div className="flex items-center gap-3 mb-6">
        <div className={`w-10 h-10 ${riskBg} rounded-lg flex items-center justify-center`}>
          <Icon className={`w-5 h-5 ${riskText}`} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Risk Assessment</h2>
          <p className="text-sm text-slate-600">Analysis Results</p>
        </div>
      </div>

      {/* Risk Level Badge */}
      <div className={`${riskBg} ${riskBorder} border-2 rounded-xl p-4 mb-6`}>
        <div className="flex items-center justify-between mb-2">
          <span className={`text-sm font-semibold ${riskText} uppercase tracking-wide`}>
            {riskLevel} RISK
          </span>
          <span className={`text-2xl font-bold ${riskText}`}>{probPercent}%</span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full bg-gradient-to-r ${
              riskColor === 'red'
                ? 'from-red-500 to-red-600'
                : riskColor === 'amber'
                ? 'from-amber-500 to-amber-600'
                : 'from-green-500 to-green-600'
            } transition-all duration-1000 ease-out`}
            style={{ width: `${probPercent}%` }}
          />
        </div>
      </div>

      {/* Prediction Details */}
      <div className="space-y-4">
        <div className="bg-slate-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-700">Default Probability</span>
            <TrendingUp className="w-4 h-4 text-slate-400" />
          </div>
          <p className="text-3xl font-bold text-slate-900">{probPercent}%</p>
          <p className="text-xs text-slate-500 mt-1">
            {willDefault ? 'Likely to default' : 'Low default risk'}
          </p>
        </div>

        {/* Risk Interpretation */}
        <div className="border-t border-slate-200 pt-4">
          <h4 className="text-sm font-semibold text-slate-900 mb-3">Risk Interpretation</h4>
          <div className="space-y-2 text-sm text-slate-600">
            {riskLevel === 'HIGH' && (
              <p className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-red-600 mt-0.5 flex-shrink-0" />
                <span>
                  High risk detected. Recommend manual review and additional verification before approval.
                </span>
              </p>
            )}
            {riskLevel === 'MEDIUM' && (
              <p className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                <span>
                  Moderate risk. Consider additional monitoring or conditional approval with safeguards.
                </span>
              </p>
            )}
            {riskLevel === 'LOW' && (
              <p className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                <span>
                  Low risk profile. Customer shows good payment behavior and creditworthiness.
                </span>
              </p>
            )}
          </div>
        </div>

        {/* Guidelines */}
        <div className="border-t border-slate-200 pt-4 mt-4">
          <h4 className="text-xs font-semibold text-slate-700 mb-2 uppercase tracking-wide">
            Guidelines
          </h4>
          <ul className="space-y-2 text-xs text-slate-600">
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-0.5">•</span>
              <span>Use this score alongside internal policies and manual review</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-0.5">•</span>
              <span>Scores are based on historical credit card data patterns</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-0.5">•</span>
              <span>Model accuracy: {prediction.model_accuracy ? `${(prediction.model_accuracy * 100).toFixed(1)}%` : 'N/A'}</span>
            </li>
          </ul>
        </div>

        <div className="mt-6 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-xs text-blue-800">
            <strong>Note:</strong> This tool is for analytical support only and does not constitute a lending decision on its own.
          </p>
        </div>
      </div>
    </div>
  )
}

