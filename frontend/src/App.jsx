import { useEffect, useRef, useState } from 'react'
import {
  ArrowRight,
  BookOpen,
  Check,
  ChevronDown,
  FileText,
  LoaderCircle,
  MessageCircle,
  PanelLeftClose,
  Plus,
  Search,
  Send,
  Sparkles,
  Trash2,
  UploadCloud,
  X,
} from 'lucide-react'
import { api } from './api'
import AuthScreen from './components/AuthScreen'
import AnswerText from './components/AnswerText'
import DocumentLibrary from './components/DocumentLibrary'
import WorkflowPanel from './components/WorkflowPanel'

const suggestions = [
  'Summarize the key points across my documents',
  'What are the most important dates mentioned?',
  'Compare the main findings in these PDFs',
]

export default function App() {
  const [user, setUser] = useState(null)
  const [checkingAuth, setCheckingAuth] = useState(true)
  const [documents, setDocuments] = useState([])
  const [conversations, setConversations] = useState([])
  const [conversationId, setConversationId] = useState(null)
  const [messages, setMessages] = useState([])
  const [selected, setSelected] = useState([])
  const [question, setQuestion] = useState('')
  const [answerMode, setAnswerMode] = useState('documents')
  const [busy, setBusy] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [deletingDocumentId, setDeletingDocumentId] = useState(null)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 700)
  const [libraryOpen, setLibraryOpen] = useState(false)
  const [workflowOpen, setWorkflowOpen] = useState(false)
  const fileInput = useRef(null)
  const bottom = useRef(null)

  useEffect(() => {
    api('/auth/me')
      .then(setUser)
      .catch(() => {})
      .finally(() => setCheckingAuth(false))
  }, [])
  useEffect(() => {
    if (!user) return
    Promise.all([api('/documents'), api('/conversations')])
      .then(([docs, chats]) => {
        setDocuments(docs)
        setConversations(chats)
      })
      .catch((err) => setError(err.message))
  }, [user?.id])
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  async function upload(files) {
    const pdfs = Array.from(files || []).filter((file) =>
      file.name.toLowerCase().endsWith('.pdf'),
    )
    if (!pdfs.length) {
      setError('Choose a PDF file to upload.')
      return
    }
    setError('')
    setUploading(true)
    try {
      for (const file of pdfs) {
        const form = new FormData()
        form.append('file', file)
        await api('/documents', { method: 'POST', body: form })
      }
      setDocuments(await api('/documents'))
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  async function openConversation(id) {
    setError('')
    try {
      const conversation = await api(`/conversations/${id}`)
      setConversationId(id)
      setMessages(conversation.messages)
      setLibraryOpen(false)
      setWorkflowOpen(false)
    } catch (err) {
      setError(err.message)
    }
  }

  function newChat() {
    setWorkflowOpen(false)
    setConversationId(null)
    setMessages([])
    setQuestion('')
    setError('')
    setLibraryOpen(false)
  }

  async function send(value = question) {
    const trimmed = value.trim()
    if (!trimmed || busy || !documents.length) return
    setQuestion('')
    setError('')
    setBusy(true)
    const optimistic = {
      id: crypto.randomUUID(),
      role: 'user',
      content: trimmed,
      citations: [],
    }
    setMessages((prev) => [...prev, optimistic])
    try {
      const result = await api('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: trimmed,
          conversation_id: conversationId,
          document_ids: selected,
          answer_mode: answerMode,
        }),
      })
      setConversationId(result.conversation_id)
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: result.answer,
          citations: result.citations,
        },
      ])
      setConversations(await api('/conversations'))
    } catch (err) {
      setMessages((prev) =>
        prev.filter((message) => message.id !== optimistic.id),
      )
      setQuestion(trimmed)
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  async function removeDocument(id) {
    const document = documents.find((item) => item.id === id)
    if (!document || deletingDocumentId) return
    if (
      !window.confirm(
        `Remove "${document.name}" from your documents? This also deletes its search data.`,
      )
    )
      return
    setError('')
    setDeletingDocumentId(id)
    try {
      await api(`/documents/${id}`, { method: 'DELETE' })
      setDocuments((prev) => prev.filter((doc) => doc.id !== id))
      setSelected((prev) => prev.filter((value) => value !== id))
    } catch (err) {
      setError(err.message)
    } finally {
      setDeletingDocumentId(null)
    }
  }

  async function removeConversation(id, event) {
    event.stopPropagation()
    if (!window.confirm('Delete this conversation?')) return
    try {
      await api(`/conversations/${id}`, { method: 'DELETE' })
      setConversations((prev) => prev.filter((chat) => chat.id !== id))
      if (conversationId === id) newChat()
    } catch (err) {
      setError(err.message)
    }
  }

  async function signOut() {
    try {
      await api('/auth/signout', { method: 'POST' })
    } finally {
      window.location.reload()
    }
  }

  if (checkingAuth)
    return (
      <div className="auth-page">
        <LoaderCircle className="spin" size={26} />
      </div>
    )
  if (!user)
    return <AuthScreen onAuthenticated={() => window.location.reload()} />

  const scopeLabel =
    selected.length === 0
      ? 'All documents'
      : `${selected.length} selected ${selected.length === 1 ? 'document' : 'documents'}`
  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="brand">
          <div className="brand-mark">
            <BookOpen size={21} strokeWidth={2.1} />
          </div>
          <span>PAGESUTRA</span>
        </div>
        <button className="new-chat" onClick={newChat}>
          <Plus size={17} /> New conversation
        </button>
        <div className="side-section-title">WORKSPACE</div>
        <button
          className={`side-link ${libraryOpen ? 'active' : ''}`}
          onClick={() => {
            setLibraryOpen(true)
            setWorkflowOpen(false)
          }}
        >
          <FileText size={17} /> My documents{' '}
          <span className="count">{documents.length}</span>
        </button>
        <button
          className={`side-link ${workflowOpen ? 'active' : ''}`}
          onClick={() => {
            setWorkflowOpen(true)
            setLibraryOpen(false)
          }}
        >
          <Sparkles size={17} /> Document workflows
        </button>
        <div className="side-section-title recent-title">RECENT CHATS</div>
        <div className="conversation-list">
          {conversations.length ? (
            conversations.map((chat) => (
              <button
                key={chat.id}
                className={`conversation-item ${conversationId === chat.id && !libraryOpen && !workflowOpen ? 'active' : ''}`}
                onClick={() => openConversation(chat.id)}
              >
                <MessageCircle size={16} />
                <span>{chat.title}</span>
                <Trash2
                  className="delete-chat"
                  size={15}
                  onClick={(event) => removeConversation(chat.id, event)}
                />
              </button>
            ))
          ) : (
            <p className="side-empty">Your conversations will appear here.</p>
          )}
        </div>
        <div className="sidebar-bottom">
          <div className="avatar">{user.email[0].toUpperCase()}</div>
          <div>
            <strong title={user.email}>{user.email}</strong>
            <small>Personal workspace</small>
          </div>
          <button className="signout-button" onClick={signOut} title="Sign out">
            Sign out
          </button>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-button toggle"
              aria-label="Toggle sidebar"
              onClick={() => setSidebarOpen((value) => !value)}
            >
              <PanelLeftClose size={19} />
            </button>
            <span className="breadcrumb">Workspace</span>
            <span className="slash">/</span>
            <strong>
              {workflowOpen
                ? 'Document workflows'
                : libraryOpen
                  ? 'My documents'
                  : conversationId
                    ? 'Conversation'
                    : 'New conversation'}
            </strong>
          </div>
          <div className="topbar-right">
            <span className="status-dot" /> Ready to explore
          </div>
        </header>
        {error && (
          <div className="error-banner">
            <span>{error}</span>
            <button aria-label="Dismiss error" onClick={() => setError('')}>
              <X size={16} />
            </button>
          </div>
        )}

        {workflowOpen ? (
          <WorkflowPanel documents={documents} selected={selected} />
        ) : libraryOpen ? (
          <DocumentLibrary
            documents={documents}
            deletingDocumentId={deletingDocumentId}
            onUpload={() => fileInput.current?.click()}
            onRemove={removeDocument}
          />
        ) : (
          <>
            <div className="chat-scroll">
              {messages.length === 0 ? (
                <div className="welcome">
                  <div className="welcome-icon">
                    <Sparkles size={26} strokeWidth={1.7} />
                  </div>
                  <div className="eyebrow">YOUR DOCUMENTS, DECODED</div>
                  <h1>
                    Answers are in the <em>details.</em>
                  </h1>
                  <p>
                    Upload your PDFs and ask anything. PAGESUTRA finds the
                    relevant passages and shows exactly where each answer came
                    from.
                  </p>
                  <div
                    className="welcome-upload"
                    onDragOver={(event) => {
                      event.preventDefault()
                      setDragging(true)
                    }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={(event) => {
                      event.preventDefault()
                      setDragging(false)
                      upload(event.dataTransfer.files)
                    }}
                  >
                    <div className="upload-icon">
                      <UploadCloud size={22} />
                    </div>
                    <div>
                      <strong>
                        {dragging
                          ? 'Drop PDFs here'
                          : uploading
                            ? 'Processing your PDF…'
                            : documents.length
                              ? `${documents.length} ${documents.length === 1 ? 'PDF' : 'PDFs'} ready to explore`
                              : 'Start with a PDF'}
                      </strong>
                      <span>
                        {documents.length
                          ? 'Add more documents or ask a question below'
                          : 'Drag and drop a file here, or browse your device'}
                      </span>
                    </div>
                    <button
                      disabled={uploading}
                      onClick={() => fileInput.current?.click()}
                    >
                      {uploading ? (
                        <LoaderCircle className="spin" size={16} />
                      ) : (
                        'Browse files'
                      )}
                    </button>
                  </div>
                  {documents.length > 0 && (
                    <div className="suggestions">
                      <span>TRY ASKING</span>
                      {suggestions.map((item) => (
                        <button key={item} onClick={() => send(item)}>
                          {item}
                          <ArrowRight size={15} />
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="messages">
                  {messages.map((message) => (
                    <div className={`message ${message.role}`} key={message.id}>
                      {message.role === 'assistant' && (
                        <div className="message-avatar">
                          <Sparkles size={16} />
                        </div>
                      )}
                      <div className="message-body">
                        <div className="message-role">
                          {message.role === 'assistant' ? 'PAGESUTRA' : 'You'}
                        </div>
                        {message.role === 'assistant' ? (
                          <AnswerText
                            text={message.content}
                            citations={message.citations || []}
                          />
                        ) : (
                          <div className="user-text">{message.content}</div>
                        )}
                        {message.citations?.length > 0 && (
                          <div className="source-list">
                            <div className="source-heading">SOURCES</div>
                            {message.citations.map((source) => (
                              <a
                                key={source.number}
                                href={`/api/documents/${source.document_id}/file#page=${source.page}`}
                                target="_blank"
                                rel="noreferrer"
                                className="source-card"
                              >
                                <span className="source-number">
                                  {source.number}
                                </span>
                                <span className="source-content">
                                  <strong>{source.document_name}</strong>
                                  <small>
                                    Page {source.page} · {source.snippet}
                                  </small>
                                </span>
                                <ArrowRight size={15} />
                              </a>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {busy && (
                    <div className="thinking">
                      <div className="message-avatar">
                        <Sparkles size={16} />
                      </div>
                      <span>
                        <LoaderCircle className="spin" size={15} /> Looking
                        through your documents…
                      </span>
                    </div>
                  )}
                  <div ref={bottom} />
                </div>
              )}
            </div>
            {documents.length > 0 && (
              <div className="chat-documents">
                <span>UPLOADED PDFS</span>
                <div className="chat-document-list">
                  {documents.map((doc) => (
                    <div className="chat-document" key={doc.id}>
                      <FileText size={14} />
                      <span title={doc.name}>{doc.name}</span>
                      <button
                        onClick={() => removeDocument(doc.id)}
                        disabled={deletingDocumentId !== null}
                        aria-label={`Remove ${doc.name}`}
                        title={`Remove ${doc.name}`}
                      >
                        {deletingDocumentId === doc.id ? (
                          <LoaderCircle className="spin" size={14} />
                        ) : (
                          <X size={14} />
                        )}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div className="composer-wrap">
              <div className="composer">
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault()
                      send()
                    }
                  }}
                  placeholder={
                    documents.length
                      ? 'Ask a question about your documents…'
                      : 'Upload a PDF to start asking questions…'
                  }
                  disabled={!documents.length || busy}
                  rows={1}
                />
                <div className="composer-bottom">
                  <label className="answer-mode">
                    <span>Answer with</span>
                    <select
                      aria-label="Answer knowledge sources"
                      value={answerMode}
                      onChange={(event) => setAnswerMode(event.target.value)}
                      disabled={busy}
                    >
                      <option value="documents">PDF evidence only</option>
                      <option value="background">
                        PDFs + related background
                      </option>
                    </select>
                  </label>
                  <button
                    className="scope-button"
                    onClick={() => setLibraryOpen(true)}
                  >
                    <FileText size={15} /> {scopeLabel}{' '}
                    <ChevronDown size={14} />
                  </button>
                  <span>Shift + Enter for new line</span>
                  <button
                    className="send-button"
                    disabled={!question.trim() || busy || !documents.length}
                    onClick={() => send()}
                    aria-label="Send message"
                  >
                    <Send size={17} />
                  </button>
                </div>
              </div>
              <div className="composer-note">
                {answerMode === 'background'
                  ? 'Related background comes from model knowledge. PDF citations apply only to the document answer.'
                  : 'PAGESUTRA can make mistakes. Check the cited pages for important details.'}
              </div>
            </div>
          </>
        )}
      </main>

      {libraryOpen && documents.length > 0 && (
        <aside className="selection-panel">
          <div className="selection-title">
            <Search size={17} /> Search scope
          </div>
          <p>
            Choose which documents to include in your next question. Leave all
            unchecked to search everything.
          </p>
          {documents.map((doc) => (
            <label key={doc.id} className="selection-row">
              <input
                type="checkbox"
                checked={selected.includes(doc.id)}
                onChange={() =>
                  setSelected((prev) =>
                    prev.includes(doc.id)
                      ? prev.filter((id) => id !== doc.id)
                      : [...prev, doc.id],
                  )
                }
              />
              <span className="custom-check">
                <Check size={12} />
              </span>
              <span title={doc.name}>{doc.name}</span>
            </label>
          ))}
          <button
            className="selection-done"
            onClick={() => setLibraryOpen(false)}
          >
            Back to chat <ArrowRight size={16} />
          </button>
        </aside>
      )}
      <input
        ref={fileInput}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        hidden
        onChange={(event) => upload(event.target.files)}
      />
    </div>
  )
}
