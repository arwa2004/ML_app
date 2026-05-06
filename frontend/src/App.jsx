import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const API_BASE = 'https://ml-patient-backend.onrender.com';

function App() {
  const [metadata, setMetadata] = useState({ genders: [], blood_types: [], conditions: [], admission_types: [] });
  const [task, setTask] = useState('diagnostic');
  const [formData, setFormData] = useState({ Age: '45', Gender: '', Blood_Type: '', Medical_Condition: '', Admission_Type: '', Stay_Duration: '7' });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    axios.get(`${API_BASE}/metadata`).then(res => setMetadata(res.data)).catch(err => console.error(err));
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const ageVal = parseInt(formData.Age);
    if (!formData.Age || isNaN(ageVal) || ageVal < 0 || ageVal > 120) {
      setError("Âge invalide (0-120)"); return;
    }
    if (!formData.Gender || !formData.Blood_Type || !formData.Medical_Condition || !formData.Admission_Type || !formData.Stay_Duration) {
      setError("Veuillez remplir tous les champs"); return;
    }
    setLoading(true); setError("");
    try {
      const response = await axios.post(`${API_BASE}/predict`, { ...formData, task });
      setResult(response.data);
    } catch (err) {
      setError("Erreur serveur");
    } finally {
      setLoading(false);
    }
  };

  const getClusterColor = (cluster) => {
    const colors = ['#10b981', '#f59e0b', '#ef4444'];
    return colors[cluster] || '#4f46e5';
  };

  return (
    <div className="app-wrapper">
      <aside className="sidebar">
        <header className="app-header"><h1>ML PATIENT APP</h1></header>
        <nav className="nav-pills">
          <button className={`nav-btn ${task === 'diagnostic' ? 'active' : ''}`} onClick={() => { setTask('diagnostic'); setResult(null); }}>
            <div style={{ fontSize: '0.6rem', opacity: 0.7 }}>PREDICTION</div>
            <div style={{ marginTop: '2px' }}>XGBOOST</div>
          </button>
          <button className={`nav-btn ${task === 'cost' ? 'active' : ''}`} onClick={() => { setTask('cost'); setResult(null); }}>
            <div style={{ fontSize: '0.6rem', opacity: 0.7 }}>RECOMMANDATION</div>
            <div style={{ marginTop: '2px' }}>RANDOM FOREST</div>
          </button>
          <button className={`nav-btn ${task === 'segmentation' ? 'active' : ''}`} onClick={() => { setTask('segmentation'); setResult(null); }}>
            <div style={{ fontSize: '0.6rem', opacity: 0.7 }}>SEGMENTATION</div>
            <div style={{ marginTop: '2px' }}>K-MEANS</div>
          </button>
        </nav>
        <div className="input-section">
          {error && <div style={{ color: '#ef4444', fontSize: '0.7rem', fontWeight: 800, marginBottom: '10px' }}>⚠️ {error}</div>}
          <form onSubmit={handleSubmit}>
            <div className="form-field"><label>AGE</label><input type="number" value={formData.Age} onChange={e => setFormData({ ...formData, Age: e.target.value })} /></div>
            <div className="form-field"><label>STAY DURATION</label><input type="number" value={formData.Stay_Duration} onChange={e => setFormData({ ...formData, Stay_Duration: e.target.value })} /></div>
            <div className="form-field"><label>GENDER</label><select value={formData.Gender} onChange={e => setFormData({ ...formData, Gender: e.target.value })}><option value="">Select...</option>{metadata.genders.map(g => <option key={g} value={g}>{g}</option>)}</select></div>
            <div className="form-field"><label>BLOOD TYPE</label><select value={formData.Blood_Type} onChange={e => setFormData({ ...formData, Blood_Type: e.target.value })}><option value="">Select...</option>{metadata.blood_types.map(b => <option key={b} value={b}>{b}</option>)}</select></div>
            <div className="form-field"><label>CONDITION</label><select value={formData.Medical_Condition} onChange={e => setFormData({ ...formData, Medical_Condition: e.target.value })}><option value="">Select...</option>{metadata.conditions.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
            <div className="form-field"><label>ADMISSION</label><select value={formData.Admission_Type} onChange={e => setFormData({ ...formData, Admission_Type: e.target.value })}><option value="">Select...</option>{metadata.admission_types.map(a => <option key={a} value={a}>{a}</option>)}</select></div>
            <button type="submit" className="action-btn" disabled={loading}>{loading ? "PROCESSING..." : "GENERATE REPORT"}</button>
          </form>
        </div>
      </aside>

      <main className="main-view">
        {result ? (
          <div className="report-container" style={{ textAlign: 'center', border: 'none', borderRadius: '24px' }}>
            <p style={{ fontWeight: 800, color: '#94a3b8', fontSize: '0.7rem', letterSpacing: '2px' }}>{result.type.toUpperCase()}</p>
            <h2 style={{ fontSize: '3.5rem', fontWeight: 900, color: getClusterColor(result.cluster), marginBottom: '20px' }}>{result.result}</h2>

            {task === 'segmentation' && (
              <>
                <div style={{ height: '350px', background: '#f8fafc', borderRadius: '16px', padding: '10px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 40, bottom: 20, left: 40 }}>
                      <XAxis type="number" dataKey="x" domain={[-8, 8]} hide />
                      <YAxis type="number" dataKey="y" domain={[-8, 8]} hide />
                      <ZAxis type="number" range={[3000, 3000]} />
                      <Scatter name="Patient" data={[{ x: result.pca_x, y: result.pca_y }]} fill={getClusterColor(result.cluster)}>
                        <Cell
                          fill={getClusterColor(result.cluster)}
                          strokeWidth={20}
                          stroke="rgba(255,255,255,1)"
                          style={{ filter: 'drop-shadow(0 4px 12px rgba(0,0,0,0.15))' }}
                        />
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ display: 'flex', marginTop: '20px', height: '10px', borderRadius: '5px', overflow: 'hidden', border: '1px solid #e2e8f0' }}>
                  <div style={{ flex: 1, background: '#10b981', opacity: result.cluster === 0 ? 1 : 0.05 }}></div>
                  <div style={{ flex: 1, background: '#f59e0b', opacity: result.cluster === 1 ? 1 : 0.05 }}></div>
                  <div style={{ flex: 1, background: '#ef4444', opacity: result.cluster === 2 ? 1 : 0.05 }}></div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', marginTop: '8px', fontWeight: 'bold', color: '#64748b' }}>
                  <span>LOW RISK</span><span>MID RISK</span><span>HIGH RISK</span>
                </div>
              </>
            )}

            {task === 'diagnostic' && (
              <div style={{ marginTop: '30px', padding: '30px', background: '#f8fafc', borderRadius: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: 800 }}>Confidence Score</span>
                  <span style={{ fontSize: '0.8rem', fontWeight: 800 }}>{result.confidence}%</span>
                </div>
                <div style={{ height: '10px', background: '#e2e8f0', borderRadius: '5px', overflow: 'hidden' }}>
                  <div style={{ width: `${result.confidence}%`, height: '100%', background: '#0284c7', transition: '1.5s ease-out' }}></div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div style={{ textAlign: 'center', opacity: 0.2 }}><h2 style={{ fontSize: '3rem', fontWeight: 900 }}>SYSTEM READY</h2></div>
        )}
      </main>
    </div>
  );
}

export default App;
