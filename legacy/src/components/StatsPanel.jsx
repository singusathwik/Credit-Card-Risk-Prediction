import { TrendingUp, Target, Zap } from 'lucide-react'

export default function StatsPanel({ accuracy }) {
  const accuracyPercent = (accuracy * 100).toFixed(2)
  
  return (
    <div className="glass rounded-2xl p-6 shadow-xl mb-8">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
            <Target className="w-7 h-7 text-white" />
          </div>
          <div>
            <p className="text-sm text-slate-600 mb-1">Model Accuracy</p>
            <p className="text-3xl font-bold gradient-text">{accuracyPercent}%</p>
          </div>
        </div>
        
        <div className="flex gap-6">
          <div className="text-center">
            <div className="flex items-center gap-2 text-slate-600 mb-1">
              <TrendingUp className="w-4 h-4" />
              <span className="text-xs font-medium">Precision</span>
            </div>
            <p className="text-lg font-semibold text-slate-900">High</p>
          </div>
          
          <div className="text-center">
            <div className="flex items-center gap-2 text-slate-600 mb-1">
              <Zap className="w-4 h-4" />
              <span className="text-xs font-medium">Speed</span>
            </div>
            <p className="text-lg font-semibold text-slate-900">Real-time</p>
          </div>
        </div>
      </div>
    </div>
  )
}

