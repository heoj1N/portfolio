import { Repository } from '../lib/github'
import Link from 'next/link'

interface RepositoryCardProps {
  repo: Repository
  readmeExcerpt?: string | null
}

export function RepositoryCard({ repo, readmeExcerpt }: RepositoryCardProps) {
  return (
    <div className="group relative flex flex-col items-start p-4 border border-neutral-200 dark:border-neutral-800 rounded-lg hover:border-neutral-300 dark:hover:border-neutral-700 transition-colors">
      <div className="flex items-center justify-between w-full mb-2">
        <Link
          href={repo.html_url}
          className="font-medium text-neutral-900 dark:text-neutral-100 hover:underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          {repo.name}
        </Link>
        <div className="flex items-center space-x-2 text-sm text-neutral-500 dark:text-neutral-400">
          <span>⭐ {repo.stargazers_count}</span>
          {repo.language && (
            <span className="flex items-center">
              <span className="w-3 h-3 rounded-full bg-current mr-1" />
              {repo.language}
            </span>
          )}
        </div>
      </div>
      
      {repo.description && (
        <p className="text-sm text-neutral-600 dark:text-neutral-300 mb-2">
          {repo.description}
        </p>
      )}
      
      {readmeExcerpt && (
        <p className="text-sm text-neutral-500 dark:text-neutral-400 mb-2 line-clamp-2">
          {readmeExcerpt}
        </p>
      )}
      
      {repo.topics.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {repo.topics.map((topic) => (
            <span
              key={topic}
              className="px-2 py-1 text-xs rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300"
            >
              {topic}
            </span>
          ))}
        </div>
      )}
    </div>
  )
} 