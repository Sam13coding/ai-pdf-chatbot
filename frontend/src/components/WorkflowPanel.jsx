import { useEffect, useState } from 'react'
import { ArrowRight, LoaderCircle, RefreshCw } from 'lucide-react'
import { api } from '../api'
import AnswerText from './AnswerText'

const modes = {
  research: 'Research PDFs',
  summary: 'Summarize documents',
  extract: 'Extract structured data',
}

export default function WorkflowPanel({ documents, selected }) {
  const [kind, setKind] = useState('research')
  const [goal, setGoal] = useState('')
  const [scope, setScope] = useState(
    selected.length ? selected : documents.map((doc) => doc.id),
  )
  const [runs, setRuns] = useState([])
  const [active, setActive] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function refresh() {
    setRuns(await api('/workflows'))
  }

  useEffect(() => {
    refresh().catch((err) => setError(err.message))
  }, [])

  async function perform(operation) {
    setBusy(true)
    setError('')
    try {
      setActive(await operation())
      await refresh()
    } catch (err) {
      setError(err.message)
      // A disconnected request may still have created a durable run.
      await refresh().catch(() => {})
    } finally {
      setBusy(false)
    }
  }

  function start(event) {
    event.preventDefault()
    perform(() =>
      api('/workflows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind, goal, document_ids: scope }),
      }),
    )
  }

  function decide(approved) {
    perform(() =>
      api(`/workflows/${active.id}/decision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved }),
      }),
    )
  }

  async function download() {
    setError('')
    try {
      const report = await api(`/workflows/${active.id}/report`)
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(report, null, 2)], {
          type: 'application/json',
        }),
      )
      const link = document.createElement('a')
      link.href = url
      link.download = `pagesutra-report-${active.id}.json`
      link.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err.message)
    }
  }

  async function removeRun() {
    if (
      !window.confirm(
        'Delete this workflow, its saved report, and checkpoint history?',
      )
    )
      return
    setBusy(true)
    setError('')
    try {
      await api(`/workflows/${active.id}`, { method: 'DELETE' })
      setActive(null)
      await refresh()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="workflow-panel">
      <div className="workflow-heading">
        <div>
          <div className="eyebrow">FROM DOCUMENTS TO DELIVERABLES</div>
          <h1>Document workflows</h1>
          <p>
            Research, summarize, or extract data from your PDFs. Review the
            evidence before saving a private report.
          </p>
        </div>
        <button
          className="workflow-secondary"
          disabled={busy}
          onClick={() => refresh().catch((err) => setError(err.message))}
          aria-label="Refresh workflows"
        >
          <RefreshCw size={16} /> Refresh
        </button>
      </div>
      {error && (
        <div className="workflow-error" role="alert">
          {error}
        </div>
      )}
      <div className="workflow-columns">
        <section>
          <form className="workflow-form" onSubmit={start}>
            <h2>Start a workflow</h2>
            <label htmlFor="workflow-kind">What would you like to do?</label>
            <select
              id="workflow-kind"
              value={kind}
              onChange={(event) => setKind(event.target.value)}
              disabled={busy}
            >
              {Object.entries(modes).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
            <label htmlFor="workflow-goal">Goal or fields to extract</label>
            <textarea
              id="workflow-goal"
              value={goal}
              onChange={(event) => setGoal(event.target.value)}
              placeholder={
                kind === 'extract'
                  ? 'Extract invoice number, total, currency, and due date.'
                  : 'Compare the main findings and identify evidence gaps.'
              }
              rows={4}
              maxLength={2000}
              required
              disabled={busy}
            />
            <fieldset disabled={busy}>
              <legend>Source documents</legend>
              {!documents.length && <p>Upload a PDF in My documents first.</p>}
              {documents.map((doc) => (
                <label className="workflow-document" key={doc.id}>
                  <input
                    type="checkbox"
                    checked={scope.includes(doc.id)}
                    onChange={() =>
                      setScope((prev) =>
                        prev.includes(doc.id)
                          ? prev.filter((id) => id !== doc.id)
                          : [...prev, doc.id],
                      )
                    }
                  />
                  {doc.name}
                </label>
              ))}
            </fieldset>
            <p className="workflow-help">
              Research searches selected PDFs. Summary and extraction read up to
              24 passages. Runs may take several minutes and use your model API.
            </p>
            <button
              className="workflow-primary"
              disabled={busy || goal.trim().length < 2 || !scope.length}
            >
              {busy ? (
                <LoaderCircle size={16} className="spin" />
              ) : (
                <ArrowRight size={16} />
              )}
              {busy ? 'Working through your documents...' : 'Start workflow'}
            </button>
          </form>
          <div className="workflow-runs">
            <h2>Saved runs</h2>
            {!runs.length && (
              <p>Your drafts and completed reports will appear here.</p>
            )}
            {runs.map((run) => (
              <button
                key={run.id}
                disabled={busy}
                className={active?.id === run.id ? 'active' : ''}
                onClick={() => perform(() => api(`/workflows/${run.id}`))}
              >
                <strong>{run.goal}</strong>
                <span>
                  {modes[run.kind]} · {run.status}
                </span>
              </button>
            ))}
          </div>
        </section>
        <section className="workflow-result" aria-live="polite">
          {active ? (
            <>
              <div className="workflow-status">{active.status}</div>
              <h2>{active.goal}</h2>
              {active.status !== 'running' && (
                <button
                  className="workflow-secondary"
                  disabled={busy}
                  onClick={removeRun}
                >
                  Delete run
                </button>
              )}
              <ol className="workflow-steps">
                {active.steps.map((step, index) => (
                  <li key={index}>{step}</li>
                ))}
              </ol>
              {active.error && (
                <p role="alert" className="workflow-error">
                  {active.error}
                </p>
              )}
              {active.draft && (
                <AnswerText text={active.draft} citations={active.citations} />
              )}
              {active.citations.length > 0 && (
                <div className="workflow-sources">
                  <h3>Source pages</h3>
                  {active.citations.map((source) => (
                    <a
                      key={source.number}
                      href={`/api/documents/${source.document_id}/file#page=${source.page}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      [{source.number}] {source.document_name} · Page{' '}
                      {source.page}
                    </a>
                  ))}
                </div>
              )}
              {active.records.length > 0 && (
                <div className="workflow-table-wrap">
                  <table className="workflow-table">
                    <thead>
                      <tr>
                        <th>Field</th>
                        <th>Value</th>
                        <th>Sources</th>
                      </tr>
                    </thead>
                    <tbody>
                      {active.records.map((record, index) => (
                        <tr key={index}>
                          <td>{record.field}</td>
                          <td>{record.value}</td>
                          <td>
                            {record.sources.map((n) => `[${n}]`).join(' ')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {active.feedback && (
                <p className="workflow-help">
                  Evidence review: {active.feedback}
                </p>
              )}
              {active.status === 'paused' && (
                <div className="workflow-approval">
                  <h3>Review before saving</h3>
                  <p>
                    Check the cited pages. Approving saves this report to your
                    private workspace.
                  </p>
                  <button
                    className="workflow-primary"
                    disabled={busy}
                    onClick={() => decide(true)}
                  >
                    Approve and save
                  </button>
                  <button
                    className="workflow-secondary"
                    disabled={busy}
                    onClick={() => decide(false)}
                  >
                    Reject
                  </button>
                </div>
              )}
              {active.can_retry && (
                <button
                  className="workflow-primary"
                  disabled={busy}
                  onClick={() =>
                    perform(() =>
                      api(`/workflows/${active.id}/retry`, { method: 'POST' }),
                    )
                  }
                >
                  Retry from checkpoint
                </button>
              )}
              {active.status === 'running' && (
                <button
                  className="workflow-secondary"
                  disabled={busy}
                  onClick={() => perform(() => api(`/workflows/${active.id}`))}
                >
                  Check progress
                </button>
              )}
              {active.status === 'blocked' && (
                <p>
                  The evidence check failed after one revision. Change the goal
                  or selected documents and start again.
                </p>
              )}
              {active.status === 'completed' && (
                <button className="workflow-primary" onClick={download}>
                  Download JSON report
                </button>
              )}
            </>
          ) : (
            <div className="workflow-empty">
              <h2>A clear path from evidence to report</h2>
              <p>
                Choose your documents and a goal. Your draft, source pages, and
                review steps will appear here.
              </p>
              <p>
                Runs persist across refreshes. Reports are saved only after your
                approval.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
