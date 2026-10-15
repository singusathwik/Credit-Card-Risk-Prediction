import { useState } from 'react'
import Header from './components/Header'
import RiskForm from './components/RiskForm'
import RiskResults from './components/RiskResults'
import StatsPanel from './components/StatsPanel'
import { Shield, TrendingUp, AlertTriangle } from 'lucide-react'

function App() {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [modelAccuracy, setModelAccuracy] = useState(null)

  const handlePredict = async (formData) => {
    setLoading(true)
    setError(null)
    
    try {
      // Fetch model accuracy first
      const accuracyRes = await fetch('/api/accuracy')
      if (accuracyRes.ok) {
        const accData = await accuracyRes.json()
        setModelAccuracy(accData.accuracy || accData.accuracy_percent / 100)
      }

      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        throw new Error('Prediction failed. Please try again.')
      }

      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      setError(err.message)
      setPrediction(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen">
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-full text-sm font-medium mb-4">
            <Shield className="w-4 h-4" />
            AI-Powered Risk Assessment
          </div>
          <h1 className="text-5xl font-bold mb-4 gradient-text">
            Credit Card Score Risk Insights
          </h1>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto">
            Advanced machine learning models analyze customer profiles to predict default risk with precision
          </p>
        </div>

        {/* Stats Panel */}
        {modelAccuracy && (
          <StatsPanel accuracy={modelAccuracy} />
        )}

        <div className="grid lg:grid-cols-3 gap-8 mt-8">
          {/* Form Section */}
          <div className="lg:col-span-2">
            <RiskForm 
              onSubmit={handlePredict} 
              loading={loading}
              error={error}
            />
          </div>

          {/* Results Section */}
          <div className="lg:col-span-1">
            <RiskResults 
              prediction={prediction}
              loading={loading}
              error={error}
            />
          </div>
        </div>

        {/* Info Cards */}
        <div className="grid md:grid-cols-3 gap-6 mt-12">
          <div className="glass rounded-2xl p-6 shadow-xl">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
              <Shield className="w-6 h-6 text-blue-600" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Advanced ML Models</h3>
            <p className="text-slate-600 text-sm">
              Powered by ensemble models including Random Forest, Gradient Boosting, and XGBoost for maximum accuracy.
            </p>
          </div>

          <div className="glass rounded-2xl p-6 shadow-xl">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Real-Time Analysis</h3>
            <p className="text-slate-600 text-sm">
              Get instant risk assessments with detailed probability scores and actionable insights.
            </p>
          </div>

          <div className="glass rounded-2xl p-6 shadow-xl">
            <div className="w-12 h-12 bg-amber-100 rounded-lg flex items-center justify-center mb-4">
              <AlertTriangle className="w-6 h-6 text-amber-600" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Risk Mitigation</h3>
            <p className="text-slate-600 text-sm">
              Identify high-risk customers early and make informed decisions to protect your portfolio.
            </p>
          </div>
        </div>
      </main>

      <footer className="mt-20 py-8 border-t border-slate-200 bg-white/50">
        <div className="container mx-auto px-4 text-center text-slate-600">
          <p className="text-sm">
            © 2024 Credit Card Score Risk Insights. Powered by Advanced Machine Learning.
          </p>
          <p className="text-xs mt-2 text-slate-500">
            This tool is for analytical support only and does not constitute a lending decision on its own.
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App

