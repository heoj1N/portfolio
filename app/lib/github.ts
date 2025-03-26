// TODO: Add proper error handling and rate limiting
// TODO: Add caching mechanism for API responses
// TODO: Add pagination support for repositories

const GITHUB_API_BASE = 'https://api.github.com'

export interface Repository {
  name: string
  description: string | null
  html_url: string
  stargazers_count: number
  language: string | null
  topics: string[]
  updated_at: string
  default_branch: string
}

export async function getRepositories(username: string): Promise<Repository[]> {
  const response = await fetch(`${GITHUB_API_BASE}/users/${username}/repos?sort=updated&per_page=100`, {
    headers: {
      'Accept': 'application/vnd.github.v3+json',
      ...(process.env.GITHUB_TOKEN && { 'Authorization': `token ${process.env.GITHUB_TOKEN}` })
    }
  })

  if (!response.ok) {
    throw new Error(`Failed to fetch repositories: ${response.statusText}`)
  }

  return response.json()
}

export async function getReadmeContent(username: string, repo: string): Promise<string | null> {
  try {
    const response = await fetch(
      `${GITHUB_API_BASE}/repos/${username}/${repo}/readme`,
      {
        headers: {
          'Accept': 'application/vnd.github.v3.raw',
          ...(process.env.GITHUB_TOKEN && { 'Authorization': `token ${process.env.GITHUB_TOKEN}` })
        }
      }
    )

    if (!response.ok) {
      return null
    }

    const content = await response.text()
    // Extract first paragraph or first few lines as excerpt
    const excerpt = content.split('\n').slice(0, 3).join('\n')
    return excerpt
  } catch (error) {
    console.error(`Failed to fetch README for ${repo}:`, error)
    return null
  }
} 