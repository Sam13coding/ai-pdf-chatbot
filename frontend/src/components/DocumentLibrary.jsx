import {
  ArrowRight,
  FilePlus2,
  FileText,
  LoaderCircle,
  Plus,
  Trash2,
} from 'lucide-react'

export default function DocumentLibrary({
  documents,
  deletingDocumentId,
  onUpload,
  onRemove,
}) {
  return (
    <section className="library-page">
      <div className="library-header">
        <div>
          <div className="eyebrow">YOUR KNOWLEDGE BASE</div>
          <h1>
            My documents<span className="accent-dot">.</span>
          </h1>
          <p>
            Keep your sources in one place. Every answer links back to its page.
          </p>
        </div>
        <button className="primary-button" onClick={() => onUpload()}>
          <Plus size={17} /> Add PDF
        </button>
      </div>
      <div className="library-grid">
        {documents.map((doc) => (
          <article className="document-card" key={doc.id}>
            <div className="document-icon">
              <FileText size={23} />
            </div>
            <div className="document-name" title={doc.name}>
              {doc.name}
            </div>
            <div className="document-meta">
              {doc.pages} {doc.pages === 1 ? 'page' : 'pages'} · {doc.chunks}{' '}
              sections
            </div>
            <div className="document-actions">
              <a
                href={`/api/documents/${doc.id}/file`}
                target="_blank"
                rel="noreferrer"
              >
                Open PDF <ArrowRight size={14} />
              </a>
              <button
                className="remove-document"
                onClick={() => onRemove(doc.id)}
                disabled={deletingDocumentId !== null}
                aria-label={`Remove ${doc.name}`}
              >
                {deletingDocumentId === doc.id ? (
                  <LoaderCircle className="spin" size={14} />
                ) : (
                  <Trash2 size={14} />
                )}{' '}
                {deletingDocumentId === doc.id ? 'Removing…' : 'Remove PDF'}
              </button>
            </div>
          </article>
        ))}
        <button className="add-card" onClick={() => onUpload()}>
          <span>
            <FilePlus2 size={24} />
          </span>
          <strong>Add another PDF</strong>
          <small>Build your knowledge base</small>
        </button>
      </div>
    </section>
  )
}
