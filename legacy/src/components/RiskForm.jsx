import { useState } from 'react'
import { Loader2, Send, Info } from 'lucide-react'

export default function RiskForm({ onSubmit, loading, error }) {
  const [formData, setFormData] = useState({
    LIMIT_BAL: '',
    SEX: '',
    EDUCATION: '',
    MARRIAGE: '',
    AGE: '',
    PAY_0: '',
    PAY_2: '',
    PAY_3: '',
    PAY_4: '',
    PAY_5: '',
    PAY_6: '',
    BILL_AMT1: '',
    BILL_AMT2: '',
    BILL_AMT3: '',
    BILL_AMT4: '',
    BILL_AMT5: '',
    BILL_AMT6: '',
    PAY_AMT1: '',
    PAY_AMT2: '',
    PAY_AMT3: '',
    PAY_AMT4: '',
    PAY_AMT5: '',
    PAY_AMT6: '',
  })

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    })
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit(formData)
  }

  return (
    <div className="glass rounded-2xl p-8 shadow-xl">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
          <Info className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Customer Profile</h2>
          <p className="text-sm text-slate-600">Enter customer information for risk assessment</p>
        </div>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Basic Information */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-900 border-b border-slate-200 pb-2">
            Basic Information
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Credit Limit (NT$)
              </label>
              <input
                type="number"
                name="LIMIT_BAL"
                value={formData.LIMIT_BAL}
                onChange={handleChange}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Gender</label>
              <select
                name="SEX"
                value={formData.SEX}
                onChange={handleChange}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="">Select</option>
                <option value="1">Male</option>
                <option value="2">Female</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Education</label>
              <select
                name="EDUCATION"
                value={formData.EDUCATION}
                onChange={handleChange}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="">Select</option>
                <option value="1">Graduate School</option>
                <option value="2">University</option>
                <option value="3">High School</option>
                <option value="4">Others</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Marital Status</label>
              <select
                name="MARRIAGE"
                value={formData.MARRIAGE}
                onChange={handleChange}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="">Select</option>
                <option value="1">Married</option>
                <option value="2">Single</option>
                <option value="3">Others</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Age</label>
              <input
                type="number"
                name="AGE"
                value={formData.AGE}
                onChange={handleChange}
                min="18"
                max="100"
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              />
            </div>
          </div>
        </div>

        {/* Repayment Status */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-900 border-b border-slate-200 pb-2">
            Repayment Status (Last 6 Months)
          </h3>
          <p className="text-xs text-slate-500 bg-blue-50 p-3 rounded-lg">
            -1 = Paid duly, 0 = Revolving credit, 1-9 = Months of delay
          </p>
          <div className="grid md:grid-cols-3 gap-4">
            {[
              { name: 'PAY_0', label: 'September' },
              { name: 'PAY_2', label: 'August' },
              { name: 'PAY_3', label: 'July' },
              { name: 'PAY_4', label: 'June' },
              { name: 'PAY_5', label: 'May' },
              { name: 'PAY_6', label: 'April' },
            ].map((field) => (
              <div key={field.name}>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  {field.label}
                </label>
                <input
                  type="number"
                  name={field.name}
                  value={formData[field.name]}
                  onChange={handleChange}
                  min="-2"
                  max="9"
                  className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>
            ))}
          </div>
        </div>

        {/* Bill Amounts */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-900 border-b border-slate-200 pb-2">
            Bill Amounts (NT$)
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            {[
              { name: 'BILL_AMT1', label: 'September' },
              { name: 'BILL_AMT2', label: 'August' },
              { name: 'BILL_AMT3', label: 'July' },
              { name: 'BILL_AMT4', label: 'June' },
              { name: 'BILL_AMT5', label: 'May' },
              { name: 'BILL_AMT6', label: 'April' },
            ].map((field) => (
              <div key={field.name}>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  {field.label}
                </label>
                <input
                  type="number"
                  name={field.name}
                  value={formData[field.name]}
                  onChange={handleChange}
                  step="100"
                  className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>
            ))}
          </div>
        </div>

        {/* Payment Amounts */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-900 border-b border-slate-200 pb-2">
            Previous Payments (NT$)
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            {[
              { name: 'PAY_AMT1', label: 'September' },
              { name: 'PAY_AMT2', label: 'August' },
              { name: 'PAY_AMT3', label: 'July' },
              { name: 'PAY_AMT4', label: 'June' },
              { name: 'PAY_AMT5', label: 'May' },
              { name: 'PAY_AMT6', label: 'April' },
            ].map((field) => (
              <div key={field.name}>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  {field.label}
                </label>
                <input
                  type="number"
                  name={field.name}
                  value={formData[field.name]}
                  onChange={handleChange}
                  step="100"
                  className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 px-6 rounded-lg font-semibold shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Analyzing Risk...</span>
            </>
          ) : (
            <>
              <Send className="w-5 h-5" />
              <span>Run Risk Assessment</span>
            </>
          )}
        </button>
      </form>
    </div>
  )
}

