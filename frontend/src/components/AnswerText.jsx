import React from 'react'

export default function AnswerText({ text, citations }) {
  const parts = text.split(/(\[\d+\])/g)
  return (
    <div className="answer-text">
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/)
        if (!match) return <React.Fragment key={i}>{part}</React.Fragment>
        const citation = citations.find((c) => c.number === Number(match[1]))
        return citation ? (
          <a
            key={i}
            className="inline-cite"
            target="_blank"
            rel="noreferrer"
            href={`/api/documents/${citation.document_id}/file#page=${citation.page}`}
          >
            {part}
          </a>
        ) : (
          <React.Fragment key={i}>{part}</React.Fragment>
        )
      })}
    </div>
  )
}
