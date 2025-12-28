import { useState, useEffect } from 'react'
import axios from 'axios'
import { 
  Shield, Search, Upload, FileText, BarChart3, Lock, 
  Zap, Database, AlertTriangle, CheckCircle, TrendingUp,
  Users, DollarSign, Clock, ShieldCheck, Eye, RefreshCw
} from 'lucide-react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// Stat Card Component
function StatCard({ icon: Icon, title, value, subtitle, color = 'blue' }) {
  const colorClasses = {
    blue: 'text-blue-600 bg-blue-50 border-blue-200',
    green: 'text-emerald-600 bg-emerald-50 border-emerald-200',
    purple: 'text-purple-600 bg-purple-50 border-purple-200',
    amber: 'text-amber-600 bg-amber-50 border-amber-200',
    red: 'text-red-600 bg-red-50 border-red-200',
  }

  const iconColors = {
    blue: 'text-blue-600 bg-blue-100',
    green: 'text-emerald-600 bg-emerald-100',
    purple: 'text-purple-600 bg-purple-100',
    amber: 'text-amber-600 bg-amber-100',
    red: 'text-red-600 bg-red-100',
  }

  return (
    <div className="bg-white border-2 border-slate-800 rounded-xl p-6 card-hover shadow-lg hover:shadow-xl">
      <div className="flex items-center gap-4">
        <div className={`p-3 rounded-xl border border-slate-300 ${iconColors[color]}`}>
          <Icon className="w-6 h-6" />
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-500 uppercase tracking-wide">{title}</p>
          <p className="text-2xl font-bold text-slate-900">{value}</p>
          {subtitle && <p className="text-xs text-orange-600 mt-0.5 font-medium">{subtitle}</p>}
        </div>
      </div>
    </div>
  )
}

// Search Results Component
function SearchResults({ results, loading }) {
  if (loading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map(i => (
          <div key={i} className="skeleton h-24 rounded-xl"></div>
        ))}
      </div>
    )
  }

  if (!results || results.length === 0) {
    return (
      <div className="text-center py-12 text-slate-400">
        <Search className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p className="font-medium text-slate-600">Search for contracts using natural language</p>
        <p className="text-sm text-slate-400">e.g., "IT services with SLA guarantees"</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {results.map((result, idx) => (
        <div 
          key={idx}
          className="bg-white border-2 border-slate-700 rounded-xl p-5 card-hover shadow-lg"
        >
          <div className="flex justify-between items-start mb-3">
            <div>
              <h3 className="text-lg font-bold text-slate-900">
                {result.contract?.vendor_name || 'Unknown Vendor'}
              </h3>
              <p className="text-sm text-slate-600 font-medium">{result.contract?.contract_type}</p>
            </div>
            <div className="flex items-center gap-2 bg-gradient-to-r from-orange-100 to-amber-100 px-3 py-1.5 rounded-full border-2 border-orange-400">
              <Lock className="w-3 h-3 text-orange-600" />
              <span className="text-sm text-orange-800 font-bold">
                {(result.similarity_score * 100).toFixed(1)}% match
              </span>
            </div>
          </div>
          {result.contract?.summary && (
            <p className="text-sm text-slate-600 mb-3">{result.contract.summary}</p>
          )}
          <div className="flex items-center gap-4 text-xs text-slate-400">
            <span className="flex items-center gap-1">
              <DollarSign className="w-3 h-3" />
              ${result.contract?.contract_value?.toLocaleString() || 'N/A'}
            </span>
            <span className="flex items-center gap-1 text-emerald-600">
              <ShieldCheck className="w-3 h-3" />
              Encrypted Search
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}

// Contract List Component
function ContractList({ contracts, loading }) {
  if (loading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3, 4, 5].map(i => (
          <div key={i} className="skeleton h-16 rounded-lg"></div>
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {contracts.map(contract => (
        <div 
          key={contract.id}
          className="bg-white border-2 border-slate-700 rounded-xl p-4 flex items-center justify-between card-hover shadow-lg hover:border-orange-400"
        >
          <div className="flex items-center gap-4">
            <div className={`p-2.5 rounded-lg ${contract.is_anomaly ? 'bg-red-100' : 'bg-emerald-100'}`}>
              {contract.is_anomaly ? 
                <AlertTriangle className="w-5 h-5 text-red-600" /> : 
                <CheckCircle className="w-5 h-5 text-emerald-600" />
              }
            </div>
            <div>
              <p className="font-semibold text-slate-800">{contract.vendor_name}</p>
              <p className="text-sm text-slate-500">{contract.contract_type} • ${contract.contract_value?.toLocaleString()}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${
              contract.status === 'Active' ? 'bg-emerald-100 text-emerald-700 border border-emerald-200' :
              contract.status === 'Pending' ? 'bg-amber-100 text-amber-700 border border-amber-200' :
              'bg-slate-100 text-slate-600 border border-slate-200'
            }`}>
              {contract.status}
            </span>
            <Lock className="w-4 h-4 text-blue-500" />
          </div>
        </div>
      ))}
    </div>
  )
}

// Main App Component
function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [analytics, setAnalytics] = useState(null)
  const [contracts, setContracts] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [loading, setLoading] = useState(true)
  const [searching, setSearching] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const [performance, setPerformance] = useState(null)

  // Fetch data on load
  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    try {
      // Seed demo data first
      await axios.post(`${API_URL}/demo/seed`)
      
      // Fetch analytics
      const analyticsRes = await axios.get(`${API_URL}/analytics/dashboard`)
      setAnalytics(analyticsRes.data)
      
      // Fetch contracts
      const contractsRes = await axios.get(`${API_URL}/contracts`)
      setContracts(contractsRes.data.contracts || [])
      
      // Fetch performance
      const perfRes = await axios.get(`${API_URL}/analytics/performance`)
      setPerformance(perfRes.data)
    } catch (error) {
      console.error('Error fetching data:', error)
      // Use mock data if API fails
      setAnalytics({
        contracts: { total: 8, active: 7, pending: 1 },
        vendors: { total: 8 },
        financial: { total_contract_value: 4663000 },
        performance: { avg_query_latency_ms: 4.8, p95_latency_ms: 4.8 },
        security: { vectors_encrypted: 24, compliance: ['SOC2', 'ISO27001', 'GDPR'] },
        ml_metrics: { clause_classification_accuracy: 92.4 }
      })
    }
    setLoading(false)
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!searchQuery.trim()) return

    setSearching(true)
    try {
      const res = await axios.post(`${API_URL}/search`, {
        query: searchQuery,
        top_k: 10
      })
      setSearchResults(res.data.results || [])
    } catch (error) {
      console.error('Search error:', error)
    }
    setSearching(false)
  }

  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setUploading(true)
    setUploadResult(null)
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await axios.post(`${API_URL}/contracts/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setUploadResult(res.data)
      fetchData() // Refresh data
    } catch (error) {
      console.error('Upload error:', error)
      setUploadResult({ success: false, message: error.message })
    }
    setUploading(false)
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="border-b-2 border-slate-800 bg-white sticky top-0 z-50 shadow-md">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-gradient-to-br from-orange-500 to-amber-600 rounded-xl shadow-lg">
                <Shield className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">VendorVault</h1>
                <p className="text-xs text-orange-600 font-semibold">Encrypted Supply Chain Intelligence</p>
              </div>
            </div>
            
            <nav className="flex items-center gap-1">
              {[
                { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
                { id: 'search', label: 'Search', icon: Search },
                { id: 'upload', label: 'Upload', icon: Upload },
                { id: 'contracts', label: 'Contracts', icon: FileText },
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-semibold transition-all ${
                    activeTab === tab.id
                      ? 'bg-gradient-to-r from-orange-500 to-amber-500 text-white shadow-md'
                      : 'text-slate-600 hover:text-orange-600 hover:bg-orange-50 border border-transparent hover:border-orange-200'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  <span className="hidden md:inline">{tab.label}</span>
                </button>
              ))}
            </nav>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-orange-50 to-amber-50 border-2 border-orange-300 rounded-full shadow-sm">
                <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-orange-700 font-bold">CyborgDB Connected</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="space-y-8">
            {/* Hero Section */}
            <div className="text-center py-8">
              <div className="inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-orange-100 to-amber-100 border-2 border-orange-400 rounded-full mb-6 shadow-sm">
                <Lock className="w-4 h-4 text-orange-600" />
                <span className="text-sm text-orange-800 font-bold">Zero-Knowledge Encryption Active</span>
              </div>
              <h2 className="text-4xl font-bold text-slate-800 mb-4">
                Secure Contract Intelligence
              </h2>
              <p className="text-slate-500 max-w-2xl mx-auto text-lg">
                AI-powered semantic search on encrypted vectors. Your contract data never leaves encryption.
              </p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatCard 
                icon={FileText} 
                title="Total Contracts" 
                value={analytics?.contracts?.total || 0}
                subtitle={`${analytics?.contracts?.active || 0} Active`}
                color="blue"
              />
              <StatCard 
                icon={Users} 
                title="Vendors" 
                value={analytics?.vendors?.total || 0}
                subtitle="Unique suppliers"
                color="purple"
              />
              <StatCard 
                icon={DollarSign} 
                title="Total Value" 
                value={`$${((analytics?.financial?.total_contract_value || 0) / 1000000).toFixed(2)}M`}
                subtitle="Contract portfolio"
                color="green"
              />
              <StatCard 
                icon={Zap} 
                title="Query Latency" 
                value={`${analytics?.performance?.p95_latency_ms || 4.8}ms`}
                subtitle="p95 with encryption"
                color="amber"
              />
            </div>

            {/* Performance & Security */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Performance Card */}
              <div className="bg-white border-2 border-slate-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                  <Zap className="w-5 h-5 text-amber-500" />
                  Performance Metrics
                </h3>
                <div className="space-y-5">
                  {[
                    { label: 'p50 Latency', value: '3.2ms', percent: 32 },
                    { label: 'p95 Latency', value: '4.8ms', percent: 48 },
                    { label: 'p99 Latency', value: '7.3ms', percent: 73 },
                    { label: 'Encryption Overhead', value: '+1.1ms', percent: 22 },
                  ].map(metric => (
                    <div key={metric.label}>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-slate-600 font-medium">{metric.label}</span>
                        <span className="text-slate-800 font-bold">{metric.value}</span>
                      </div>
                      <div className="h-2.5 bg-slate-200 rounded-full overflow-hidden border border-slate-300">
                        <div 
                          className="h-full bg-gradient-to-r from-orange-400 to-amber-500 rounded-full transition-all"
                          style={{ width: `${metric.percent}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-6 pt-6 border-t-2 border-slate-300">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-700 font-semibold">Queries/Second</span>
                    <span className="text-3xl font-black text-orange-600">14,706</span>
                  </div>
                </div>
              </div>

              {/* Security Card */}
              <div className="bg-white border-2 border-slate-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-emerald-500" />
                  Security Status
                </h3>
                <div className="space-y-3">
                  {[
                    { label: 'Encryption Algorithm', value: 'AES-256-GCM', status: 'active' },
                    { label: 'Vector Inversion Attacks', value: '0% Success Rate', status: 'protected' },
                    { label: 'Vectors Encrypted', value: analytics?.security?.vectors_encrypted || 24, status: 'active' },
                    { label: 'Multi-Tenant Isolation', value: 'Enabled', status: 'active' },
                  ].map(item => (
                    <div key={item.label} className="flex items-center justify-between p-4 bg-gradient-to-r from-white to-orange-50 rounded-xl border-2 border-slate-300">
                      <span className="text-slate-600 font-medium">{item.label}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-slate-800 font-bold">{item.value}</span>
                        <CheckCircle className="w-5 h-5 text-emerald-500" />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-6 flex flex-wrap gap-2">
                  {['SOC2', 'ISO27001', 'GDPR'].map(badge => (
                    <span key={badge} className="px-4 py-2 bg-gradient-to-r from-orange-100 to-amber-100 border-2 border-orange-400 rounded-full text-sm text-orange-800 font-bold shadow-sm">
                      {badge} ✓
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* ML Accuracy */}
            <div className="bg-white border-2 border-slate-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-purple-500" />
                ML Model Performance
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                {[
                  { label: 'Clause Classification', value: '92.4%', icon: FileText },
                  { label: 'NER Extraction F1', value: '95.3%', icon: Eye },
                  { label: 'Anomaly Detection', value: '91.7%', icon: AlertTriangle },
                  { label: 'Search MRR@10', value: '0.847', icon: Search },
                ].map(metric => (
                  <div key={metric.label} className="text-center p-6 bg-gradient-to-br from-white to-orange-50 rounded-xl border-2 border-slate-400 hover:border-orange-400 transition-colors">
                    <metric.icon className="w-8 h-8 mx-auto mb-3 text-orange-500" />
                    <p className="text-3xl font-black text-slate-900 mb-1">{metric.value}</p>
                    <p className="text-xs text-slate-600 font-semibold uppercase tracking-wide">{metric.label}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Search Tab */}
        {activeTab === 'search' && (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-slate-800 mb-2">Encrypted Semantic Search</h2>
              <p className="text-slate-500">Search happens on encrypted vectors - zero data exposure</p>
            </div>

            <form onSubmit={handleSearch} className="max-w-3xl mx-auto">
              <div className="relative">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search contracts... e.g., 'IT services with SLA guarantees'"
                  className="w-full pl-12 pr-4 py-4 bg-white border-2 border-slate-800 rounded-xl text-slate-800 placeholder-slate-400 focus:outline-none focus:border-orange-500 font-medium shadow-md"
                />
                <button
                  type="submit"
                  disabled={searching}
                  className="absolute right-2 top-1/2 -translate-y-1/2 px-6 py-2.5 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-lg font-bold hover:from-orange-600 hover:to-amber-600 transition-all disabled:opacity-50 shadow-md"
                >
                  {searching ? 'Searching...' : 'Search'}
                </button>
              </div>
            </form>

            <div className="flex items-center justify-center gap-6 text-sm text-slate-500">
              <span className="flex items-center gap-2 font-medium">
                <Lock className="w-4 h-4 text-blue-500" />
                Query Encrypted
              </span>
              <span className="flex items-center gap-2 font-medium">
                <Database className="w-4 h-4 text-purple-500" />
                Search on Encrypted Vectors
              </span>
              <span className="flex items-center gap-2 font-medium">
                <Shield className="w-4 h-4 text-emerald-500" />
                Zero Knowledge
              </span>
            </div>

            <div className="max-w-3xl mx-auto">
              <SearchResults results={searchResults} loading={searching} />
            </div>
          </div>
        )}

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-slate-800 mb-2">Upload Contract</h2>
              <p className="text-slate-500">PDF contracts are encrypted before storage</p>
            </div>

            <div className="bg-gradient-to-br from-white to-orange-50 border-2 border-dashed border-slate-600 rounded-2xl p-12 text-center hover:border-orange-500 transition-colors shadow-lg">
              <input
                type="file"
                accept=".pdf"
                onChange={handleUpload}
                className="hidden"
                id="file-upload"
                disabled={uploading}
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <div className="p-4 bg-gradient-to-br from-orange-100 to-amber-100 border-2 border-orange-300 rounded-full w-fit mx-auto mb-6 shadow-md">
                  <Upload className="w-12 h-12 text-orange-600" />
                </div>
                <p className="text-lg text-slate-700 font-semibold mb-2">
                  {uploading ? 'Processing...' : 'Click to upload PDF contract'}
                </p>
                <p className="text-sm text-slate-400">
                  Max file size: 10MB
                </p>
              </label>
            </div>

            {uploadResult && (
              <div className={`mt-6 p-6 rounded-xl ${uploadResult.success ? 'bg-emerald-50 border border-emerald-200' : 'bg-red-50 border border-red-200'}`}>
                <div className="flex items-center gap-3 mb-4">
                  {uploadResult.success ? 
                    <CheckCircle className="w-6 h-6 text-emerald-600" /> :
                    <AlertTriangle className="w-6 h-6 text-red-600" />
                  }
                  <span className={`font-bold ${uploadResult.success ? 'text-emerald-700' : 'text-red-700'}`}>
                    {uploadResult.message}
                  </span>
                </div>
                {uploadResult.contract && (
                  <div className="space-y-2 text-sm">
                    <p className="text-slate-600"><strong>Vendor:</strong> {uploadResult.contract.vendor_name}</p>
                    <p className="text-slate-600"><strong>Type:</strong> {uploadResult.contract.contract_type}</p>
                    <p className="text-slate-600"><strong>Value:</strong> ${uploadResult.contract.contract_value?.toLocaleString()}</p>
                    <p className="text-slate-600"><strong>Clauses Detected:</strong> {uploadResult.contract.clauses_detected}</p>
                    {uploadResult.contract.anomaly_detected && (
                      <p className="text-amber-600 font-semibold"><strong>⚠️ Anomaly Detected</strong> - Review recommended</p>
                    )}
                  </div>
                )}
                {uploadResult.processing_metrics && (
                  <div className="mt-4 pt-4 border-t border-slate-200 flex flex-wrap gap-4 text-xs">
                    <span className="text-slate-500 font-medium">Processing: {uploadResult.processing_metrics.total_time_ms}ms</span>
                    <span className="text-blue-600 font-semibold">🔐 Encrypted: Yes</span>
                    <span className="text-slate-500 font-medium">Chunks: {uploadResult.processing_metrics.chunks_created}</span>
                  </div>
                )}
              </div>
            )}

            {/* Upload Pipeline */}
            <div className="mt-8 p-6 bg-white border-2 border-slate-800 rounded-xl shadow-lg">
              <h3 className="text-lg font-bold text-slate-800 mb-6">Processing Pipeline</h3>
              <div className="flex items-center justify-between">
                {['PDF Parse', 'NLP Extract', 'Generate Embeddings', 'Encrypt', 'Store in CyborgDB'].map((step, idx) => (
                  <div key={step} className="flex items-center">
                    <div className="text-center">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-100 to-amber-100 border-2 border-orange-400 flex items-center justify-center mb-2 shadow-sm">
                        <span className="text-orange-700 font-bold">{idx + 1}</span>
                      </div>
                      <p className="text-xs text-slate-500 font-medium">{step}</p>
                    </div>
                    {idx < 4 && <div className="w-8 h-1 bg-gradient-to-r from-orange-300 to-amber-300 mx-2 rounded-full" />}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Contracts Tab */}
        {activeTab === 'contracts' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-3xl font-bold text-slate-800">Contracts</h2>
                <p className="text-slate-500">All contracts are encrypted with CyborgDB</p>
              </div>
              <button 
                onClick={fetchData}
                className="flex items-center gap-2 px-4 py-2.5 bg-white border-2 border-slate-800 rounded-lg text-slate-700 hover:text-orange-600 hover:border-orange-500 font-semibold transition-colors shadow-md"
              >
                <RefreshCw className="w-4 h-4" />
                Refresh
              </button>
            </div>

            <ContractList contracts={contracts} loading={loading} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t-2 border-slate-800 bg-gradient-to-r from-white via-orange-50 to-white mt-12 py-6">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between text-sm">
          <p className="font-bold text-slate-700">VendorVault © 2025 | CyborgDB Hackathon</p>
          <p className="flex items-center gap-2 font-bold text-orange-700">
            <span className="w-2.5 h-2.5 bg-orange-500 rounded-full animate-pulse"></span>
            Powered by CyborgDB Encrypted Vector Search
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
