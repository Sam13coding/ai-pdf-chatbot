export async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    credentials: 'same-origin',
    ...options,
  })
  if (response.status === 401 && !path.startsWith('/auth/'))
    window.location.reload()
  if (response.status === 204) return null
  const data = await response.json().catch(() => ({}))
  if (!response.ok)
    throw new Error(data.detail || `Request failed (${response.status})`)
  return data
}
