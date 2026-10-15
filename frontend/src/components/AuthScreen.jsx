import { useState } from 'react'
import { ArrowRight, BookOpen } from 'lucide-react'
import { api } from '../api'

export default function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function submit(event) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      const user = await api(`/auth/${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      })
      onAuthenticated(user)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-brand">
          <div className="brand-mark">
            <BookOpen size={21} />
          </div>
          <div className="auth-brand-copy">
            <span>PAGESUTRA</span>
            <span className="auth-brand-tagline">AI PDF ASSISTANT</span>
          </div>
        </div>
        <div className="eyebrow">YOUR PERSONAL WORKSPACE</div>
        <h1>{mode === 'signup' ? 'Create your account.' : 'Welcome back.'}</h1>
        <p>Your PDFs, questions, and answers stay saved in your account.</p>
        <form onSubmit={submit}>
          <label>
            Email
            <input
              type="email"
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
              maxLength={254}
            />
          </label>
          <label>
            Password
            <input
              type="password"
              autoComplete={
                mode === 'signup' ? 'new-password' : 'current-password'
              }
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
              minLength={8}
              maxLength={128}
            />
          </label>
          {error && (
            <div className="auth-error" role="alert">
              {error}
            </div>
          )}
          <button className="primary-button" disabled={busy} type="submit">
            {busy ? 'Please wait…' : mode === 'signup' ? 'Sign up' : 'Sign in'}{' '}
            <ArrowRight size={16} />
          </button>
        </form>
        <div className="auth-switch">
          {mode === 'signup' ? 'Already have an account?' : 'New to PAGESUTRA?'}{' '}
          <button
            type="button"
            onClick={() => {
              setMode(mode === 'signup' ? 'signin' : 'signup')
              setError('')
            }}
          >
            {mode === 'signup' ? 'Sign in' : 'Sign up'}
          </button>
        </div>
      </div>
    </div>
  )
}
