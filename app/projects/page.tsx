import { Metadata } from 'next'
import { getRepositories, getReadmeContent } from '../lib/github'
import { RepositoryCard } from '../components/repository-card'

export const metadata: Metadata = {
  title: 'Projects',
  description: 'My open source projects and repositories.',
}

// TODO: Move to environment variable
const GITHUB_USERNAME = 'heoj1N'

export default async function ProjectsPage() {
  const repositories = await getRepositories(GITHUB_USERNAME)
  
  // Fetch README excerpts for each repository
  const reposWithReadme = await Promise.all(
    repositories.map(async (repo) => {
      const readmeExcerpt = await getReadmeContent(GITHUB_USERNAME, repo.name)
      return { ...repo, readmeExcerpt }
    })
  )

  return (
    <section>
      <h1 className="font-semibold text-2xl mb-8 tracking-tighter">Projects</h1>
      <div className="grid gap-4 sm:grid-cols-2">
        {reposWithReadme.map((repo) => (
          <RepositoryCard
            key={repo.name}
            repo={repo}
            readmeExcerpt={repo.readmeExcerpt}
          />
        ))}
      </div>
    </section>
  )
} 