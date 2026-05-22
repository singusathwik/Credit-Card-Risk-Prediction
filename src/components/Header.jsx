import { CreditCard, Sparkles } from 'lucide-react'

export default function Header() {
  return (
    <header className="glass border-b border-slate-200/50 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
              <CreditCard className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">Risk Insights</h1>
              <p className="text-xs text-slate-500">Credit Card Analytics</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-full">
            <Sparkles className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-700">AI-Powered</span>
          </div>
        </div>
      </div>
    </header>
  )
}

