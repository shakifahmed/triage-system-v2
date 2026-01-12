import React, { useState, useEffect, useMemo } from 'react';
import { Search, ChevronDown, Activity, Calendar, User, AlertCircle, Star } from 'lucide-react';

const API_URL = "/api";
// const API_URL = "http://127.0.0.1:8000";

const TriageSystem = () => {
  const [symptoms, setSymptoms] = useState([]);
  const [symptomList, setSymptomList] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [age, setAge] = useState('30');
  const [gender, setGender] = useState('Male');
  const [severity, setSeverity] = useState('Moderate');
  const [duration, setDuration] = useState('3');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState(false);
  const [error, setError] = useState('');
  const [predictionId, setPredictionId] = useState(null);

  // Feedback states
  const [rating, setRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [feedbackText, setFeedbackText] = useState('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  useEffect(() => {
    fetch('/symptom_master_list.csv')
      .then(res => res.text())
      .then(text => {
        const lines = text.split('\n').slice(1);
        const symptoms = lines
          .map(line => line.trim())
          .filter(line => line.length > 0);
        setSymptomList(symptoms);
      });
  }, []);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const resp = await fetch(`${API_URL}/health`);
        const data = await resp.json();

        // v2 API health fields: status, model_loaded, mlb_loaded, ...
        setApiStatus(Boolean(data?.status === 'healthy' && data?.model_loaded));
      } catch {
        setApiStatus(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const filteredSymptoms = useMemo(() => {
    if (!searchTerm.trim()) return [];

    const term = searchTerm.toLowerCase();
    const scored = symptomList
      .filter(s => !symptoms.includes(s))
      .map(symptom => {
        const lower = symptom.toLowerCase();
        let score = 0;

        if (lower === term) score = 1000;
        else if (lower.startsWith(term)) score = 500;
        else if (lower.includes(term)) score = 100;
        else {
          let termIdx = 0;
          for (let char of lower) {
            if (char === term[termIdx]) {
              termIdx++;
              if (termIdx === term.length) {
                score = 50;
                break;
              }
            }
          }
        }

        return { symptom, score };
      })
      .filter(item => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);

    return scored.map(item => item.symptom);
  }, [searchTerm, symptomList, symptoms]);

  const handleKeyDown = (e) => {
    if (!showDropdown || filteredSymptoms.length === 0) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev =>
        prev < filteredSymptoms.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => prev > 0 ? prev - 1 : -1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (selectedIndex >= 0) {
        addSymptom(filteredSymptoms[selectedIndex]);
      }
    } else if (e.key === 'Escape') {
      setShowDropdown(false);
      setSelectedIndex(-1);
    }
  };

  const addSymptom = (symptom) => {
    if (!symptoms.includes(symptom)) {
      setSymptoms([...symptoms, symptom]);
    }
    setSearchTerm('');
    setShowDropdown(false);
    setSelectedIndex(-1);
  };

  const removeSymptom = (symptom) => {
    setSymptoms(symptoms.filter(s => s !== symptom));
  };

  const clearAllSymptoms = () => {
    setSymptoms([]);
    setSearchTerm('');
    setShowDropdown(false);
    setSelectedIndex(-1);
  };

  const handleAgeChange = (e) => {
    const value = e.target.value;
    // Allow empty string or valid numbers
    if (value === '' || /^\d+$/.test(value)) {
      setAge(value);
    }
  };

  const handleDurationChange = (e) => {
    const value = e.target.value;
    // Allow empty string or valid numbers
    if (value === '' || /^\d+$/.test(value)) {
      setDuration(value);
    }
  };

  const isFormValid = () => {
    return (
      symptoms.length > 0 &&
      age !== '' &&
      parseInt(age) > 0 &&
      duration !== '' &&
      parseInt(duration) >= 0 &&
      gender !== '' &&
      severity !== ''
    );
  };

  const handleSubmit = async () => {
    if (!isFormValid()) {
      setError('Please fill in all required fields');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    setRating(0);
    setFeedbackText('');
    setFeedbackSubmitted(false);
    setPredictionId(null);

    const payload = {
      symptoms: symptoms.join(', '),
      age: parseInt(age),
      gender,
      severity,
      duration: parseInt(duration)
    };

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      setPredictionId(data.prediction_id);
    } catch (err) {
      setError(err.message || 'Failed to get prediction');
    } finally {
      setLoading(false);
    }
  };

  const handleFeedbackSubmit = async () => {
    if (!rating && !feedbackText.trim()) {
      setError('Please provide either a rating or comments');
      return;
    }

    setFeedbackLoading(true);
    setError('');

    // v2 API feedback expects rating + feedback_text
    const feedbackPayload = {
      rating: rating || null,
      feedback_text: feedbackText.trim() || null
      // If your backend accepts/needs prediction_id, you can add it back:
      // prediction_id: predictionId
    };

    try {
      const response = await fetch(`${API_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedbackPayload)
      });

      if (!response.ok) {
        throw new Error(`Failed to submit feedback: ${response.status}`);
      }

      setFeedbackSubmitted(true);
      setRating(0);
      setFeedbackText('');
      setTimeout(() => {
        setFeedbackSubmitted(false);
      }, 3000);
    } catch (err) {
      setError(err.message || 'Failed to submit feedback');
    } finally {
      setFeedbackLoading(false);
    }
  };

  const ratingLabels = ['Bad', 'Poor', 'Good', 'Very Good', 'Excellent'];

  // Helpers to match v2 API fields + your UI wording
  const getDisplayRecommendation = (apiRec) => (apiRec === 'Drug' ? 'OTC Drug' : apiRec);
  const isDoctor = (apiRec) => apiRec === 'Doctor Consultation';

  return (
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-cyan-50 via-blue-50 to-purple-50">
      <div className="fixed inset-0 opacity-30">
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-200 via-blue-200 to-purple-200 animate-gradient"></div>
      </div>

      <div className="relative z-10">
        <header className="px-6 py-6">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Activity className="w-8 h-8 text-cyan-600" />
              <h1 className="text-2xl font-bold text-gray-800">Primary Triage</h1>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${apiStatus ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-600">{apiStatus ? 'Online' : 'Offline'}</span>
            </div>
          </div>
        </header>

        <div className="max-w-7xl mx-auto px-6 pt-12 pb-16">
          <div className="text-center mb-16">
            <h2 className="text-6xl font-bold mb-4 text-gray-900">
              Empowering health<br />
              <span className="text-gray-700">through deep AI</span>
            </h2>
            <p className="text-xl text-gray-600 mt-4">
              AI-powered triage for informed healthcare decisions
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto md:items-center">
            <div className="bg-white/80 backdrop-blur-lg rounded-3xl shadow-xl p-8 border border-white/20 hover:shadow-2xl transition-all duration-300">
              <h3 className="text-2xl font-bold text-gray-800 mb-6">Patient Assessment</h3>

              <div className="space-y-6">
                <div className="relative">
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Symptoms *
                  </label>
                  <div className="relative">
                    <Search className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
                    <input
                      type="text"
                      value={searchTerm}
                      onChange={(e) => {
                        setSearchTerm(e.target.value);
                        setShowDropdown(true);
                        setSelectedIndex(-1);
                      }}
                      onFocus={() => setShowDropdown(true)}
                      onKeyDown={handleKeyDown}
                      placeholder="Search symptoms..."
                      className="w-full pl-10 pr-20 py-3 rounded-xl border border-gray-200 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-200 transition-all outline-none"
                    />
                    {symptoms.length > 0 && (
                      <button
                        type="button"
                        onClick={clearAllSymptoms}
                        className="absolute right-3 top-3 text-xs text-cyan-600 hover:text-cyan-700 font-semibold transition-colors"
                      >
                        Clear all
                      </button>
                    )}
                  </div>

                  {showDropdown && filteredSymptoms.length > 0 && (
                    <div className="absolute z-50 w-full mt-2 bg-white rounded-xl shadow-2xl border border-gray-100 max-h-64 overflow-y-auto">
                      {filteredSymptoms.map((symptom, idx) => (
                        <div
                          key={idx}
                          onClick={() => addSymptom(symptom)}
                          className={`px-4 py-3 cursor-pointer transition-colors border-b border-gray-50 last:border-0 ${
                            idx === selectedIndex ? 'bg-cyan-100' : 'hover:bg-cyan-50'
                          }`}
                        >
                          {symptom}
                        </div>
                      ))}
                    </div>
                  )}

                  {symptoms.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {symptoms.map((symptom, idx) => (
                        <div
                          key={idx}
                          className="inline-flex items-center bg-cyan-100 text-cyan-800 px-3 py-1.5 rounded-full text-sm font-medium"
                        >
                          <span className="max-w-xs truncate">{symptom}</span>
                          <button
                            type="button"
                            onClick={() => removeSymptom(symptom)}
                            className="ml-2 hover:text-cyan-900 font-bold"
                          >
                            ×
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      <User className="inline w-4 h-4 mr-1" />
                      Age *
                    </label>
                    <input
                      type="text"
                      value={age}
                      onChange={handleAgeChange}
                      placeholder="Enter age"
                      className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-200 transition-all outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Gender *
                    </label>
                    <div className="relative">
                      <select
                        value={gender}
                        onChange={(e) => setGender(e.target.value)}
                        className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-200 transition-all outline-none appearance-none bg-white"
                      >
                        <option>Male</option>
                        <option>Female</option>
                      </select>
                      <ChevronDown className="absolute right-3 top-3.5 w-5 h-5 text-gray-400 pointer-events-none" />
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      <AlertCircle className="inline w-4 h-4 mr-1" />
                      Severity *
                    </label>
                    <div className="relative">
                      <select
                        value={severity}
                        onChange={(e) => setSeverity(e.target.value)}
                        className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-200 transition-all outline-none appearance-none bg-white"
                      >
                        <option>Mild</option>
                        <option>Moderate</option>
                        <option>Severe</option>
                      </select>
                      <ChevronDown className="absolute right-3 top-3.5 w-5 h-5 text-gray-400 pointer-events-none" />
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      <Calendar className="inline w-4 h-4 mr-1" />
                      Duration (days) *
                    </label>
                    <input
                      type="text"
                      value={duration}
                      onChange={handleDurationChange}
                      placeholder="Enter days"
                      className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-200 transition-all outline-none"
                    />
                  </div>
                </div>

                {error && (
                  <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl">
                    {error}
                  </div>
                )}

                <button
                  onClick={handleSubmit}
                  disabled={loading || !apiStatus || !isFormValid()}
                  className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 text-white py-4 rounded-xl font-semibold hover:from-cyan-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl"
                >
                  {loading ? 'Analyzing...' : 'Get Recommendation'}
                </button>
              </div>
            </div>

            <div className="bg-white/80 backdrop-blur-lg rounded-3xl shadow-xl p-8 border border-white/20 hover:shadow-2xl transition-all duration-300">
              <h3 className="text-2xl font-bold text-gray-800 mb-6">AI Analysis</h3>

              {!result ? (
                <div className="flex flex-col items-center justify-center h-64 text-center">
                  <Activity className="w-16 h-16 text-gray-300 mb-4" />
                  <p className="text-gray-500">Submit patient information to see AI recommendation</p>
                </div>
              ) : (
                <div className="space-y-5">
                  {/* Recommendation with confidence + explanation */}
                  <div className={`p-6 rounded-2xl relative ${
                    isDoctor(result.recommendation)
                      ? 'bg-gradient-to-br from-red-50 to-orange-50 border border-red-200'
                      : 'bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200'
                  }`}>
                    {/* Confidence badge in top-right (v2: confidence_percent already 0..100) */}
                    <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm px-3 py-1.5 rounded-full border border-blue-200 shadow-sm">
                      <div className="flex items-center space-x-1.5">
                        <span className="text-xs font-semibold text-gray-600">CONFIDENCE</span>
                        <span className="text-sm font-bold text-blue-600">
                          {Number(result.confidence_percent ?? 0).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div className="text-sm font-semibold text-gray-600 mb-2">RECOMMENDATION</div>
                    <div className="text-3xl font-bold text-gray-900 mb-3 pr-24">
                      {getDisplayRecommendation(result.recommendation)}
                    </div>

                    {/* Explanation from API (LLM/user_explanation) */}
                    {result.user_explanation && (
                      <p className="text-gray-700 text-sm leading-relaxed whitespace-pre-line">
                        {result.user_explanation}
                      </p>
                    )}
                  </div>

                  {/* Feedback Section (smaller + compact) - ID removed */}
                  <div className="bg-gray-50 p-4 rounded-xl border border-gray-200">
                    <div className="mb-2">
                      <h4 className="text-sm font-bold text-gray-800">
                        We value your opinion
                      </h4>
                      <p className="text-xs text-gray-600">
                        Rating & comments
                      </p>
                    </div>

                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center">
                        {[1, 2, 3, 4, 5].map((star) => (
                          <button
                            key={star}
                            type="button"
                            onClick={() => setRating(star)}
                            onMouseEnter={() => setHoverRating(star)}
                            onMouseLeave={() => setHoverRating(0)}
                            className="p-1 transition-transform hover:scale-110"
                            disabled={feedbackSubmitted}
                            aria-label={`Rate ${star}`}
                          >
                            <Star
                              className={`w-6 h-6 ${
                                star <= (hoverRating || rating)
                                  ? 'fill-yellow-400 text-yellow-400'
                                  : 'text-gray-300'
                              }`}
                            />
                          </button>
                        ))}
                      </div>

                      <div className="text-xs text-gray-500 min-w-[70px] text-right">
                        {rating > 0 ? <span className="font-medium">{ratingLabels[rating - 1]}</span> : <span>—</span>}
                      </div>
                    </div>

                    <textarea
                      value={feedbackText}
                      onChange={(e) => setFeedbackText(e.target.value)}
                      placeholder="Comments..."
                      disabled={feedbackSubmitted}
                      className="mt-3 w-full px-3 py-2 rounded-xl border border-gray-200 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-200 transition-all outline-none resize-none text-sm"
                      rows="2"
                    />

                    {feedbackSubmitted ? (
                      <div className="mt-2 text-center text-green-600 font-medium text-sm">
                        Thank you!
                      </div>
                    ) : (
                      <button
                        onClick={handleFeedbackSubmit}
                        disabled={feedbackLoading || (!rating && !feedbackText.trim())}
                        className="mt-3 w-full bg-gradient-to-r from-slate-600 to-slate-700 text-white py-2.5 rounded-xl font-semibold hover:from-slate-700 hover:to-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all text-sm"
                      >
                        {feedbackLoading ? 'Submitting...' : 'Submit'}
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="mt-16 max-w-5xl mx-auto">
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-white/30">
                <h4 className="font-bold text-lg mb-2 text-gray-800">How It Works</h4>
                <p className="text-sm text-gray-600">
                  Our AI intelligently interprets patient symptoms, demographics, and severity to support timely and informed triage decisions.
                </p>
              </div>
              <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-white/30">
                <h4 className="font-bold text-lg mb-2 text-gray-800">Clinical Support</h4>
                <p className="text-sm text-gray-600">
                  This tool guides patients on whether their symptoms may need a doctor's attention or if simple OTC (Over-The-Counter) care may be enough.
                </p>
              </div>
              <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-white/30">
                <h4 className="font-bold text-lg mb-2 text-gray-800">Continuous Learning</h4>
                <p className="text-sm text-gray-600">
                  Built on medical knowledge bases and continuously refined to improve accuracy and reliability.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes gradient {
          0% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(180deg); }
          100% { transform: translateY(0) rotate(360deg); }
        }
        .animate-gradient {
          animation: gradient 20s ease infinite;
        }
      `}</style>
    </div>
  );
};

export default TriageSystem;