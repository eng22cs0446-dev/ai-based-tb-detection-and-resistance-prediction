# NexTB - Tuberculosis Detection System

NexTB is a comprehensive AI-powered Tuberculosis detection system. It features a robust Python/FastAPI backend for processing X-rays utilizing machine learning, and a sleek React/Vite frontend dashboard for medical professionals to review inferences, case histories, and detailed probability overlays.

## Project Structure

- **`backend/`**: Contains the FastAPI server, SQLite database handlers, and inference logic.
- **`frontend/`**: Contains the React + Vite single page application (SPA).
- **`tb_system/`**: Core machine learning models, attention map generation (GradCAM++), and PDF report builders.

## Dataset
Full dataset download:
https://drive.google.com/file/d/1Mfe-U-pKej5b6jsUacfu1BEOH_fxu6ht/view?usp=sharing
---

## 🛠️ Installation & Setup

If you have just cloned this repository, you will need to set up both the **Backend API** and the **Frontend Web App**.

### 1. Backend Setup (Python)

The backend handles the AI inference and data storage. You will need Python 3.9+ installed.

1. Open a terminal and navigate to the project root:
   ```bash
   cd TB_Detection_SystemV1
   ```
2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the backend server:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
   *The backend API will now be running at `http://localhost:8000/api/v1`*

### 2. Frontend Setup (Node.js/React)

The frontend provides the interactive user interface. You will need [Node.js](https://nodejs.org/) installed (which includes `npm`).

1. Open a **new** terminal window and navigate to the `frontend` folder:
   ```bash
   cd TB_Detection_SystemV1/frontend
   ```
2. Install the React dependencies (this reads from `package.json`):
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *The frontend application will now be running at `http://localhost:5173`. Open this URL in your browser to view the dashboard!*

---

## 🚀 Building for Production

If you wish to deploy the system to a production environment:

**Frontend build:**
```bash
cd frontend
npm run build
```
This will compile the optimized React assets into the `frontend/dist/` directory, which can then be served statically by engines like Nginx, Vercel, or even mounted directly to the FastAPI server.
