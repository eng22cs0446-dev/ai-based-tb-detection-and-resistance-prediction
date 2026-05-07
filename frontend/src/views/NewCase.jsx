import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/api';
import Card from '../components/Card';
import { Upload } from 'lucide-react';

export default function NewCase() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    patient_id: '',
    age: '',
    gender: 'Male',
    district: '',
    symptoms: [],
    history: [],
    other_comorbidities: ''
  });
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleCheckboxChange = (category, value) => {
    setFormData(prev => {
      const list = prev[category];
      if (list.includes(value)) {
        return { ...prev, [category]: list.filter(item => item !== value) };
      } else {
        return { ...prev, [category]: [...list, value] };
      }
    });
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setImage(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return alert('Please upload an X-ray image');
    setLoading(true);
    
    try {
      // Split base64 out from data URL
      const base64 = image.split(',')[1];
      const payload = {
        ...formData,
        age: parseInt(formData.age, 10) || 30, // Mock default if empty
      };
      
      const result = await api.predictTb(payload, base64);
      navigate(`/cases/${result.case_id}`);
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 style={{ fontSize: '2rem', marginBottom: '1.5rem', color: 'var(--color-text-primary)' }}>New Case</h2>
      
      <form onSubmit={handleSubmit} style={{ display: 'grid', gridTemplateColumns: 'minmax(350px, 1fr) 1fr', gap: '2rem' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <Card title="Patient Information">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--color-text-secondary)' }}>Patient ID</label>
                  <input required name="patient_id" value={formData.patient_id} onChange={handleInputChange} 
                         style={{ width: '100%', padding: '0.75rem', borderRadius: 'var(--radius-md)', backgroundColor: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', color: 'var(--color-text-primary)' }} />
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--color-text-secondary)' }}>Age</label>
                  <input required name="age" type="number" value={formData.age} onChange={handleInputChange} 
                         style={{ width: '100%', padding: '0.75rem', borderRadius: 'var(--radius-md)', backgroundColor: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', color: 'var(--color-text-primary)' }} />
                </div>
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--color-text-secondary)' }}>Gender</label>
                  <select name="gender" value={formData.gender} onChange={handleInputChange}
                          style={{ width: '100%', padding: '0.75rem', borderRadius: 'var(--radius-md)', backgroundColor: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', color: 'var(--color-text-primary)' }}>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--color-text-secondary)' }}>District</label>
                  <input name="district" value={formData.district} onChange={handleInputChange} 
                         style={{ width: '100%', padding: '0.75rem', borderRadius: 'var(--radius-md)', backgroundColor: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', color: 'var(--color-text-primary)' }} />
                </div>
              </div>
            </div>
          </Card>
          
          <button type="submit" disabled={loading} style={{
             width: '100%', padding: '1rem',
             backgroundColor: 'var(--color-accent)', color: 'var(--color-bg-base)',
             fontWeight: 600, borderRadius: 'var(--radius-md)',
             boxShadow: 'var(--shadow-glow)', opacity: loading ? 0.7 : 1, transition: '0.2s'
          }}>
            {loading ? 'Analyzing...' : 'Analyze Image'}
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: '100%' }}>
          <Card title="X-ray Upload" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <p style={{ color: 'var(--color-text-secondary)', marginBottom: '1rem', fontSize: '0.9rem' }}>Upload a high-resolution chest X-ray image (DICOM, JPG, PNG)</p>
            <div style={{ 
              flex: 1, border: '2px dashed var(--color-border)', borderRadius: 'var(--radius-lg)', 
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              backgroundColor: 'var(--color-bg-elevated)', position: 'relative', overflow: 'hidden'
            }}>
              {image ? (
                <img src={image} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : (
                <>
                  <Upload size={48} color="var(--color-text-muted)" style={{ marginBottom: '1rem' }} />
                  <p style={{ fontWeight: 500, color: 'var(--color-text-primary)' }}>Click to upload or drag and drop</p>
                  <p style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)', marginTop: '0.5rem' }}>Maximum file size: 20MB</p>
                </>
              )}
              <input type="file" accept="image/*" onChange={handleImageUpload} style={{ 
                opacity: 0, position: 'absolute', inset: 0, cursor: 'pointer' 
              }} />
            </div>
          </Card>
        </div>
      </form>
    </div>
  );
}
