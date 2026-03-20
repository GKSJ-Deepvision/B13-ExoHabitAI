// ExoHabitAI Frontend Application

const { useState, useEffect } = React;

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// Main App Component
function App() {
    const [activeTab, setActiveTab] = useState('binary');
    const [formData, setFormData] = useState({
        mass_earth: 1.0,
        semimajor_axis: 1.0,
        star_temp_k: 5778,
        star_luminosity: 0.0,
        star_metallicity: 0.0,
        log_stellar_flux: 0.0,
        log_surface_gravity: 0.69,
        bulk_density_gcc: 5.51,
        star_class: 'G'
    });
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [apiStatus, setApiStatus] = useState(null);

    // Check API health on mount
    useEffect(() => {
        checkApiHealth();
    }, []);

    const checkApiHealth = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/health`);
            setApiStatus(response.data);
        } catch (err) {
            setApiStatus({ status: 'offline' });
        }
    };

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: parseFloat(value) || 0
        }));
    };

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const endpoint = activeTab === 'binary' 
                ? `${API_BASE_URL}/predict/binary`
                : activeTab === 'multiclass'
                ? `${API_BASE_URL}/predict/multiclass`
                : `${API_BASE_URL}/predict/both`;

            const response = await axios.post(endpoint, formData);
            setResults(response.data);
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to get prediction');
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setFormData({
            mass_earth: 1.0,
            semimajor_axis: 1.0,
            star_temp_k: 5778,
            star_luminosity: 0.0,
            star_metallicity: 0.0,
            log_stellar_flux: 0.0,
            log_surface_gravity: 0.69,
            bulk_density_gcc: 5.51,
            star_class: 'G'
        });
        setResults(null);
        setError(null);
    };

    const loadExample = (type) => {
        const examples = {
            earth: {
                mass_earth: 1.0,
                semimajor_axis: 1.0,
                star_temp_k: 5778,
                star_luminosity: 0.0,
                star_metallicity: 0.0,
                log_stellar_flux: 0.0,
                log_surface_gravity: 0.69,
                bulk_density_gcc: 5.51,
                star_class: 'G'
            },
            hot: {
                mass_earth: 0.8,
                semimajor_axis: 0.05,
                star_temp_k: 6000,
                star_luminosity: 0.3,
                star_metallicity: 0.1,
                log_stellar_flux: 1.5,
                log_surface_gravity: 0.8,
                bulk_density_gcc: 5.2,
                star_class: 'F'
            },
            cold: {
                mass_earth: 1.2,
                semimajor_axis: 5.0,
                star_temp_k: 4500,
                star_luminosity: -0.5,
                star_metallicity: -0.2,
                log_stellar_flux: -1.5,
                log_surface_gravity: 0.6,
                bulk_density_gcc: 4.8,
                star_class: 'K'
            },
            gasGiant: {
                mass_earth: 318.0,
                semimajor_axis: 5.2,
                star_temp_k: 5778,
                star_luminosity: 0.0,
                star_metallicity: 0.0,
                log_stellar_flux: -0.7,
                log_surface_gravity: 2.5,
                bulk_density_gcc: 1.3,
                star_class: 'G'
            }
        };
        setFormData(examples[type]);
        setResults(null);
        setError(null);
    };

    return (
        <div className="app-container">
            <Header apiStatus={apiStatus} />
            
            <div className="main-content">
                <div className="card">
                    <h2 className="card-title">Input Parameters</h2>
                    
                    <ExampleButtons loadExample={loadExample} />
                    
                    <InputForm 
                        formData={formData}
                        handleInputChange={handleInputChange}
                    />
                    
                    <div className="btn-group">
                        <button 
                            className="btn btn-primary" 
                            onClick={handlePredict}
                            disabled={loading}
                        >
                            {loading ? 'Predicting...' : 'Predict Habitability'}
                        </button>
                        <button 
                            className="btn btn-secondary" 
                            onClick={handleReset}
                        >
                            Reset
                        </button>
                    </div>
                </div>
                
                <div className="card">
                    <h2 className="card-title">Prediction Results</h2>
                    
                    <Tabs activeTab={activeTab} setActiveTab={setActiveTab} />
                    
                    {loading && <Loading />}
                    {error && <Error message={error} />}
                    {results && (
                        <Results 
                            results={results} 
                            activeTab={activeTab}
                        />
                    )}
                    {!loading && !error && !results && (
                        <div className="info-box">
                            <h3>Ready to Predict</h3>
                            <p>Enter exoplanet parameters and click "Predict Habitability" to get started.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

// Header Component
function Header({ apiStatus }) {
    return (
        <div className="header">
            <h1>🌍 ExoHabitAI</h1>
            <p>Machine Learning-Powered Exoplanet Habitability Prediction</p>
            {apiStatus && (
                <div style={{ marginTop: '10px', fontSize: '0.875rem' }}>
                    <span style={{ 
                        color: apiStatus.status === 'healthy' ? '#10b981' : '#ef4444',
                        fontWeight: '600'
                    }}>
                        ● API Status: {apiStatus.status === 'healthy' ? 'Online' : 'Offline'}
                    </span>
                </div>
            )}
        </div>
    );
}

// Example Buttons Component
function ExampleButtons({ loadExample }) {
    return (
        <div className="info-box" style={{ marginBottom: '20px' }}>
            <h3>Quick Examples</h3>
            <div style={{ display: 'flex', gap: '8px', marginTop: '12px', flexWrap: 'wrap' }}>
                <button className="btn btn-secondary" style={{ padding: '8px 16px', fontSize: '0.875rem' }} onClick={() => loadExample('earth')}>
                    Earth-like
                </button>
                <button className="btn btn-secondary" style={{ padding: '8px 16px', fontSize: '0.875rem' }} onClick={() => loadExample('hot')}>
                    Hot Planet
                </button>
                <button className="btn btn-secondary" style={{ padding: '8px 16px', fontSize: '0.875rem' }} onClick={() => loadExample('cold')}>
                    Cold Planet
                </button>
                <button className="btn btn-secondary" style={{ padding: '8px 16px', fontSize: '0.875rem' }} onClick={() => loadExample('gasGiant')}>
                    Gas Giant
                </button>
            </div>
        </div>
    );
}

// Input Form Component
function InputForm({ formData, handleInputChange }) {
    const fields = [
        { name: 'mass_earth', label: 'Planet Mass', unit: 'Earth masses', step: '0.1' },
        { name: 'semimajor_axis', label: 'Semi-major Axis', unit: 'AU', step: '0.1' },
        { name: 'star_temp_k', label: 'Star Temperature', unit: 'Kelvin', step: '100' },
        { name: 'star_luminosity', label: 'Star Luminosity', unit: 'log10(L/L_sun)', step: '0.1' },
        { name: 'star_metallicity', label: 'Star Metallicity', unit: 'dex', step: '0.1' },
        { name: 'log_stellar_flux', label: 'Log Stellar Flux', unit: 'log10(Earth flux)', step: '0.1' },
        { name: 'log_surface_gravity', label: 'Log Surface Gravity', unit: 'log1p(g/g_Earth)', step: '0.1' },
        { name: 'bulk_density_gcc', label: 'Bulk Density', unit: 'g/cm³', step: '0.1' }
    ];

    const handleStarClassChange = (e) => {
        handleInputChange({ target: { name: 'star_class', value: e.target.value } });
    };

    return (
        <div>
            {fields.map(field => (
                <div key={field.name} className="form-group">
                    <label>{field.label}</label>
                    <input
                        type="number"
                        name={field.name}
                        value={formData[field.name]}
                        onChange={handleInputChange}
                        step={field.step}
                    />
                    <small>{field.unit}</small>
                </div>
            ))}
            <div className="form-group">
                <label>Star Class</label>
                <select 
                    name="star_class" 
                    value={formData.star_class} 
                    onChange={handleStarClassChange}
                    style={{ 
                        width: '100%', 
                        padding: '10px', 
                        borderRadius: '8px', 
                        border: '1px solid #e5e7eb',
                        fontSize: '14px',
                        backgroundColor: 'white'
                    }}
                >
                    <option value="A">A - Hot, Blue-White</option>
                    <option value="B">B - Very Hot, Blue</option>
                    <option value="F">F - White</option>
                    <option value="G">G - Yellow (Sun-like)</option>
                    <option value="K">K - Orange</option>
                    <option value="M">M - Red Dwarf</option>
                    <option value="Unknown">Unknown</option>
                </select>
                <small>Spectral classification of the host star</small>
            </div>
        </div>
    );
}

// Tabs Component
function Tabs({ activeTab, setActiveTab }) {
    return (
        <div className="tabs">
            <button 
                className={`tab ${activeTab === 'binary' ? 'active' : ''}`}
                onClick={() => setActiveTab('binary')}
            >
                Binary Classification
            </button>
            <button 
                className={`tab ${activeTab === 'multiclass' ? 'active' : ''}`}
                onClick={() => setActiveTab('multiclass')}
            >
                Multi-Class
            </button>
            <button 
                className={`tab ${activeTab === 'both' ? 'active' : ''}`}
                onClick={() => setActiveTab('both')}
            >
                Both
            </button>
        </div>
    );
}

// Results Component
function Results({ results, activeTab }) {
    if (activeTab === 'both') {
        return (
            <div className="results-container">
                {results.binary && <BinaryResult result={results.binary} />}
                {results.multiclass && <MultiClassResult result={results.multiclass} />}
            </div>
        );
    } else if (activeTab === 'binary') {
        return <BinaryResult result={results} />;
    } else {
        return <MultiClassResult result={results} />;
    }
}

// Binary Result Component
function BinaryResult({ result }) {
    const isHabitable = result.prediction === 1;
    
    return (
        <div className="result-card">
            <div className="result-header">
                <h3 className="result-title">Binary Classification</h3>
                <span className={`result-badge ${isHabitable ? 'badge-habitable' : 'badge-not-habitable'}`}>
                    {result.prediction_label}
                </span>
            </div>
            
            <ConfidenceMeter confidence={result.confidence} />
            
            <div className="probability-container">
                <div className="probability-item">
                    <div className="probability-label">
                        <span>Not Habitable</span>
                        <span>{(result.probability.not_habitable * 100).toFixed(1)}%</span>
                    </div>
                    <div className="probability-bar">
                        <div 
                            className="probability-fill" 
                            style={{ width: `${result.probability.not_habitable * 100}%` }}
                        />
                    </div>
                </div>
                
                <div className="probability-item">
                    <div className="probability-label">
                        <span>Habitable</span>
                        <span>{(result.probability.habitable * 100).toFixed(1)}%</span>
                    </div>
                    <div className="probability-bar">
                        <div 
                            className="probability-fill" 
                            style={{ width: `${result.probability.habitable * 100}%` }}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}

// Multi-Class Result Component
function MultiClassResult({ result }) {
    const getBadgeClass = (label) => {
        const map = {
            'Cold': 'badge-cold',
            'Rocky-Habitable': 'badge-habitable',
            'Hot': 'badge-hot',
            'Gas-Giant': 'badge-gas-giant'
        };
        return map[label] || 'badge-not-habitable';
    };
    
    return (
        <div className="result-card">
            <div className="result-header">
                <h3 className="result-title">Multi-Class Classification</h3>
                <span className={`result-badge ${getBadgeClass(result.prediction_label)}`}>
                    {result.prediction_label}
                </span>
            </div>
            
            <ConfidenceMeter confidence={result.confidence} />
            
            <div className="probability-container">
                {Object.entries(result.probabilities).map(([className, prob]) => (
                    <div key={className} className="probability-item">
                        <div className="probability-label">
                            <span>{className}</span>
                            <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="probability-bar">
                            <div 
                                className="probability-fill" 
                                style={{ width: `${prob * 100}%` }}
                            />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

// Confidence Meter Component
function ConfidenceMeter({ confidence }) {
    return (
        <div className="confidence-meter">
            <div className="confidence-label">
                Confidence: {(confidence * 100).toFixed(1)}%
            </div>
            <div className="confidence-bar">
                <div 
                    className="confidence-fill" 
                    style={{ width: `${confidence * 100}%` }}
                >
                    <span className="confidence-text">{(confidence * 100).toFixed(1)}%</span>
                </div>
            </div>
        </div>
    );
}

// Loading Component
function Loading() {
    return (
        <div className="loading">
            <div className="spinner"></div>
            <p>Analyzing exoplanet parameters...</p>
        </div>
    );
}

// Error Component
function Error({ message }) {
    return (
        <div className="error">
            <strong>Error:</strong> {message}
        </div>
    );
}

// Render App
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
