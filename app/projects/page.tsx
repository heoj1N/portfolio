import { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Projects',
  description: 'My open source projects and repositories.',
}

export default function ProjectsPage() {
  return (
    <section>
      <h1 className="font-semibold text-2xl mb-8 tracking-tighter">Projects</h1>
      <div className="prose prose-neutral dark:prose-invert">
        {/* TODO: Add GitHub API integration to fetch repositories */}
        {/* TODO: Add repository cards with README excerpts */}
        {/* TODO: Add filtering and sorting options */}
        <p>Coming soon: A showcase of my open source projects and repositories.</p>
      </div>
    </section>
  )
} 