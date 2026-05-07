const BASE_URL = 'http://localhost:8000/api/v1';

export const api = {
  // Check health
  async checkHealth() {
    const res = await fetch(`${BASE_URL}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return res.json();
  },

  // Get dashboard stats
  async getDashboardStats(days = 30) {
    const res = await fetch(`${BASE_URL}/dashboard/stats?days=${days}`);
    if (!res.ok) throw new Error('Failed to fetch dashboard stats');
    return res.json();
  },

  // Get filtered cases list
  async getCases(page = 1, limit = 20) {
    const res = await fetch(`${BASE_URL}/cases?page=${page}&limit=${limit}`);
    if (!res.ok) throw new Error('Failed to fetch cases');
    return res.json();
  },

  // Get detailed case result
  async getCaseDetails(caseId) {
    const res = await fetch(`${BASE_URL}/cases/${caseId}`);
    if (!res.ok) throw new Error('Failed to fetch case details');
    return res.json();
  },

  // Predict new X-ray
  async predictTb(patientData, base64Image) {
    const payload = {
      patient: patientData,
      xray_base64: base64Image
    };
    const res = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to predict TB');
    }
    return res.json();
  },

  // Download PDF Report link generator
  getReportDownloadUrl(caseId) {
    return `${BASE_URL}/cases/${caseId}/pdf`;
  },

  // Delete a case
  async deleteCase(caseId) {
    const res = await fetch(`${BASE_URL}/cases/${caseId}`, {
      method: 'DELETE'
    });
    if (!res.ok) throw new Error('Failed to delete case');
    return res.json();
  },

  // Get cases for a specific patient
  async getPatientCases(patientId) {
    const res = await fetch(`${BASE_URL}/patients/${patientId}/cases`);
    if (res.status === 404) return { cases: [], total: 0 }; // Handle no prior cases
    if (!res.ok) throw new Error('Failed to fetch patient cases');
    return res.json();
  }
};
