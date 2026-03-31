import React, { useCallback, useEffect, useMemo, useState } from 'react';
import './App.css';

const API_BASE = (process.env.REACT_APP_API_BASE || '').replace(/\/$/, '');

const tokenStorageKey = 'researchmind_auth';

function decodeJwtPayload(token) {
	try {
		const base64 = token.split('.')[1];
		if (!base64) return null;
		const normalized = base64.replace(/-/g, '+').replace(/_/g, '/');
		const decoded = decodeURIComponent(
			atob(normalized)
				.split('')
				.map((c) => `%${`00${c.charCodeAt(0).toString(16)}`.slice(-2)}`)
				.join('')
		);
		return JSON.parse(decoded);
	} catch {
		return null;
	}
}

function getStoredAuth() {
	try {
		const raw = localStorage.getItem(tokenStorageKey);
		if (!raw) return null;
		const parsed = JSON.parse(raw);
		if (!parsed?.access_token) return null;
		const payload = decodeJwtPayload(parsed.access_token);
		if (payload?.exp && Date.now() >= payload.exp * 1000) {
			localStorage.removeItem(tokenStorageKey);
			return null;
		}
		return parsed;
	} catch {
		return null;
	}
}

function saveAuth(authPayload) {
	localStorage.setItem(tokenStorageKey, JSON.stringify(authPayload));
}

function clearAuth() {
	localStorage.removeItem(tokenStorageKey);
}

async function apiFetch(path, options = {}, token) {
	const headers = new Headers(options.headers || {});
	if (token) headers.set('Authorization', `Bearer ${token}`);
	const response = await fetch(`${API_BASE}${path}`, {
		...options,
		headers,
	});
	if (!response.ok) {
		let detail = `Request failed: ${response.status}`;
		try {
			const data = await response.json();
			detail = data?.detail || data?.message || detail;
		} catch {
			// keep default message
		}
		throw new Error(detail);
	}
	if (response.status === 204) return null;
	return response.json();
}

function App() {
	const [authView, setAuthView] = useState('signup');
	const [health, setHealth] = useState({
		loading: true,
		ok: false,
		message: 'Checking API...',
		endpoint: '/health',
	});
	const [lastChecked, setLastChecked] = useState('never');
	const [auth, setAuth] = useState(() => getStoredAuth());
	const [authError, setAuthError] = useState('');
	const [authLoading, setAuthLoading] = useState(false);
	const [loginForm, setLoginForm] = useState({ username: '', password: '' });
	const [registerForm, setRegisterForm] = useState({
		username: '',
		password: '',
		role: 'analyst',
		email: '',
	});

	const [docs, setDocs] = useState([]);
	const [docsLoading, setDocsLoading] = useState(false);
	const [uploading, setUploading] = useState(false);
	const [uploadMessage, setUploadMessage] = useState('');
	const [selectedDocIds, setSelectedDocIds] = useState([]);

	const [question, setQuestion] = useState('');
	const [queryResult, setQueryResult] = useState(null);
	const [queryError, setQueryError] = useState('');
	const [queryLoading, setQueryLoading] = useState(false);
	const [topK, setTopK] = useState(5);

	const candidates = useMemo(
		() => [
			'/health',
			'http://127.0.0.1:8010/health',
			'http://localhost:8010/health',
			'http://127.0.0.1:8000/health',
			'http://localhost:8000/health',
		],
		[]
	);

	const authToken = auth?.access_token || '';

	const checkHealth = useCallback(async (signal) => {
		let lastError = 'No response from backend';
		for (const url of candidates) {
			try {
				const res = await fetch(url, { signal });
				if (!res.ok) {
					lastError = `${url} returned ${res.status}`;
					continue;
				}
				const data = await res.json();
				setHealth({
					loading: false,
					ok: true,
					message: `${data.status || 'ok'} (${data.environment || 'unknown'})`,
					endpoint: url,
				});
				setLastChecked(new Date().toLocaleTimeString());
				return;
			} catch (error) {
				if (error.name === 'AbortError') {
					return;
				}
				lastError = `${url} failed: ${error.message}`;
			}
		}

		setHealth({
			loading: false,
			ok: false,
			message: lastError,
			endpoint: 'No healthy endpoint found',
		});
		setLastChecked(new Date().toLocaleTimeString());
	}, [candidates]);

	const loadDocuments = useCallback(async () => {
		setDocsLoading(true);
		try {
			const data = await apiFetch('/api/v1/documents', {}, authToken);
			setDocs(data || []);
		} catch (error) {
			setUploadMessage(`Could not load documents: ${error.message}`);
		} finally {
			setDocsLoading(false);
		}
	}, [authToken]);

	const handleManualCheck = useCallback(async () => {
		setHealth((prev) => ({ ...prev, loading: true, message: 'Rechecking API...' }));
		await checkHealth();
	}, [checkHealth]);

	const handleLogin = async (event) => {
		event.preventDefault();
		setAuthLoading(true);
		setAuthError('');
		try {
			const data = await apiFetch('/api/v1/auth/login', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(loginForm),
			});
			saveAuth(data);
			setAuth(data);
			setAuthView('login');
		} catch (error) {
			setAuthError(error.message);
		} finally {
			setAuthLoading(false);
		}
	};

	const handleRegister = async (event) => {
		event.preventDefault();
		setAuthLoading(true);
		setAuthError('');
		try {
			await apiFetch('/api/v1/auth/register', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(registerForm),
			});
			setLoginForm({ username: registerForm.username, password: registerForm.password });
			setAuthView('login');
			setAuthError('Registration successful. Log in to open your dashboard.');
		} catch (error) {
			setAuthError(error.message);
		} finally {
			setAuthLoading(false);
		}
	};

	const handleLogout = () => {
		clearAuth();
		setAuth(null);
		setAuthView('login');
	};

	const handleUpload = async (event) => {
		event.preventDefault();
		const file = event.target.file.files[0];
		if (!file) {
			setUploadMessage('Select a file before upload.');
			return;
		}

		setUploading(true);
		setUploadMessage('');
		try {
			const formData = new FormData();
			formData.append('file', file);
			const data = await apiFetch('/api/v1/upload', {
				method: 'POST',
				body: formData,
			}, authToken);
			setUploadMessage(data?.message || 'Upload completed.');
			event.target.reset();
			await loadDocuments();
		} catch (error) {
			setUploadMessage(`Upload failed: ${error.message}`);
		} finally {
			setUploading(false);
		}
	};

	const handleAsk = async (event) => {
		event.preventDefault();
		setQueryLoading(true);
		setQueryError('');
		setQueryResult(null);
		try {
			const payload = {
				question,
				top_k: Number(topK) || 5,
			};
			if (selectedDocIds.length) payload.doc_ids = selectedDocIds;

			const data = await apiFetch('/api/v1/query', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(payload),
			}, authToken);
			setQueryResult(data);
		} catch (error) {
			setQueryError(error.message);
		} finally {
			setQueryLoading(false);
		}
	};

	const toggleDocSelection = (docId) => {
		setSelectedDocIds((prev) =>
			prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
		);
	};

	useEffect(() => {
		const controller = new AbortController();
		checkHealth(controller.signal);
		const timer = setInterval(() => checkHealth(controller.signal), 10000);

		const onFocus = () => checkHealth(controller.signal);
		window.addEventListener('focus', onFocus);

		return () => {
			clearInterval(timer);
			window.removeEventListener('focus', onFocus);
			controller.abort();
		};
	}, [checkHealth]);

	useEffect(() => {
		if (authToken) {
			loadDocuments();
			return;
		}
		setDocs([]);
		setSelectedDocIds([]);
	}, [authToken, loadDocuments]);

	const statusClass = useMemo(() => {
		if (health.loading) return 'status pending';
		return health.ok ? 'status good' : 'status bad';
	}, [health]);

	if (!auth) {
		return (
			<div className="app-shell">
				<div className="orb orb-one" />
				<div className="orb orb-two" />
				<main className="panel auth-wrapper">
					<p className="eyebrow">ResearchMind</p>
					<h1>Welcome to Your Document Intelligence Workspace</h1>
					<p className="lede">
						Create your account, sign in, and continue to the dashboard to analyze reports,
						company documents, policies, and PDFs with grounded citations.
					</p>

					<section className="card-row">
						<article className="card">
							<h2>Backend Health</h2>
							<div className={statusClass}>{health.message}</div>
							<div className="health-tools">
								<button type="button" onClick={handleManualCheck}>Recheck now</button>
								<span>Last checked: {lastChecked}</span>
							</div>
							<p>Active endpoint: {health.endpoint}</p>
						</article>
						<article className="card">
							<h2>Workspace Links</h2>
							<div className="actions">
								<a href="http://127.0.0.1:8010/docs" target="_blank" rel="noreferrer">Open API Reference</a>
								<a href="http://127.0.0.1:8010/health" target="_blank" rel="noreferrer">Open Health JSON</a>
							</div>
						</article>
					</section>

					<section className="card auth-panel">
						<div className="auth-switch">
							<button
								type="button"
								className={authView === 'signup' ? 'active' : ''}
								onClick={() => setAuthView('signup')}
							>
								Sign Up
							</button>
							<button
								type="button"
								className={authView === 'login' ? 'active' : ''}
								onClick={() => setAuthView('login')}
							>
								Log In
							</button>
						</div>

						{authView === 'signup' ? (
							<form className="form" onSubmit={handleRegister}>
								<h2>Create Account</h2>
								<input
									type="text"
									placeholder="Username"
									value={registerForm.username}
									onChange={(e) => setRegisterForm((p) => ({ ...p, username: e.target.value }))}
									required
								/>
								<input
									type="password"
									placeholder="Password"
									value={registerForm.password}
									onChange={(e) => setRegisterForm((p) => ({ ...p, password: e.target.value }))}
									required
								/>
								<input
									type="email"
									placeholder="Email (optional)"
									value={registerForm.email}
									onChange={(e) => setRegisterForm((p) => ({ ...p, email: e.target.value }))}
								/>
								<select
									value={registerForm.role}
									onChange={(e) => setRegisterForm((p) => ({ ...p, role: e.target.value }))}
								>
									<option value="analyst">Analyst</option>
									<option value="portfolio_manager">Portfolio Manager</option>
									<option value="compliance">Compliance</option>
									<option value="executive">Executive</option>
								</select>
								<button type="submit" disabled={authLoading}>Create account</button>
								<p className="small-note">
									Already have an account?{' '}
									<button type="button" className="link-button" onClick={() => setAuthView('login')}>
										Log in
									</button>
								</p>
							</form>
						) : (
							<form className="form" onSubmit={handleLogin}>
								<h2>Log In</h2>
								<input
									type="text"
									placeholder="Username"
									value={loginForm.username}
									onChange={(e) => setLoginForm((p) => ({ ...p, username: e.target.value }))}
									required
								/>
								<input
									type="password"
									placeholder="Password"
									value={loginForm.password}
									onChange={(e) => setLoginForm((p) => ({ ...p, password: e.target.value }))}
									required
								/>
								<button type="submit" disabled={authLoading}>Enter dashboard</button>
								<p className="small-note">
									Need an account?{' '}
									<button type="button" className="link-button" onClick={() => setAuthView('signup')}>
										Sign up
									</button>
								</p>
							</form>
						)}
						{authError ? <p className="small-note">{authError}</p> : null}
					</section>
				</main>
			</div>
		);
	}

	return (
		<div className="app-shell">
			<div className="orb orb-one" />
			<div className="orb orb-two" />
			<main className="panel">
				<div className="header-row">
					<div>
						<p className="eyebrow">ResearchMind</p>
						<h1>Main Dashboard</h1>
					</div>
					<div className="user-badge">
						<div>
							<strong>{auth.username}</strong>
							<p>{auth.role_label || auth.role}</p>
						</div>
						<button type="button" onClick={handleLogout}>Logout</button>
					</div>
				</div>
				<p className="lede">
					Analyze any uploaded corpus with secure access control, evidence-backed answers,
					and source-level citations.
				</p>

				<div className="platform-metrics" role="status" aria-live="polite">
					<span>{docs.length} documents indexed</span>
					<span>{selectedDocIds.length} selected for scope</span>
					<span>Signed in as {auth.username}</span>
				</div>

				<section className="card-row stack-2">
					<article className="card">
						<h2>Backend Health</h2>
						<div className={statusClass}>{health.message}</div>
						<div className="health-tools">
							<button type="button" onClick={handleManualCheck}>Recheck now</button>
							<span>Last checked: {lastChecked}</span>
						</div>
						<p>Active endpoint: {health.endpoint}</p>
					</article>
					<article className="card">
						<h2>Ingest Files</h2>
						<form className="form" onSubmit={handleUpload}>
							<input type="file" name="file" accept=".pdf,.txt,.docx" required />
							<button type="submit" disabled={uploading}>Upload File</button>
						</form>
						<p className="small-note">Accepted formats: PDF, TXT, DOCX.</p>
						{uploadMessage ? <p className="small-note">{uploadMessage}</p> : null}
					</article>
				</section>

				<section className="card-row stack-2">
					<article className="card">
						<h2>Source Scope</h2>
						<div className="doc-list">
							{docsLoading ? <p className="small-note">Loading documents...</p> : null}
							{!docsLoading && docs.length === 0 ? (
								<p className="small-note">No documents uploaded yet.</p>
							) : null}
							{docs.map((doc) => (
								<label key={doc.doc_id} className="doc-item">
									<input
										type="checkbox"
										checked={selectedDocIds.includes(doc.doc_id)}
										onChange={() => toggleDocSelection(doc.doc_id)}
									/>
									<span>{doc.filename}</span>
								</label>
							))}
						</div>
					</article>

					<article className="card">
						<h2>Ask the Corpus</h2>
						<form className="form" onSubmit={handleAsk}>
							<textarea
								placeholder="Ask a grounded question across uploaded reports, policies, and research documents"
								value={question}
								onChange={(e) => setQuestion(e.target.value)}
								rows={4}
								required
							/>
							<label className="small-note">
								Top K
								<input
									type="number"
									min="1"
									max="20"
									value={topK}
									onChange={(e) => setTopK(e.target.value)}
								/>
							</label>
							<button type="submit" disabled={queryLoading}>Analyze</button>
						</form>

						{queryError ? <p className="small-note">{queryError}</p> : null}

						{queryResult ? (
							<div className="result-box">
								<h3>Answer</h3>
								<p>{queryResult.answer}</p>
								{queryResult.analysis ? (
									<>
										<h3>Analysis</h3>
										<p>{queryResult.analysis}</p>
									</>
								) : null}
								<h3>Citations</h3>
								<ul>
									{(queryResult.sources || []).map((source, idx) => (
										<li key={`${source.doc_id}-${source.chunk_index}-${idx}`}>
											<strong>{source.filename}</strong> chunk {source.chunk_index} (score{' '}
											{Number(source.score).toFixed(3)})
											<div>{source.text}</div>
										</li>
									))}
								</ul>
							</div>
						) : null}
					</article>
				</section>

				<footer className="hint">
					If health is red, restart backend with: python -m uvicorn app.main:app --reload --port 8010
				</footer>
			</main>
		</div>
	);
}

export default App;
