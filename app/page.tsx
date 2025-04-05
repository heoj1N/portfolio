import { BlogPosts } from 'app/components/posts'
import { Papers } from 'app/components/papers'
import NeuralNetwork from 'app/components/NeuralNetwork'
import VortexBlackHole from 'app/components/VortexBlackHole'
import Link from 'next/link'
import { getRepositories, getReadmeContent } from 'app/lib/github'
import { RepositoryCard } from 'app/components/repository-card'

// TODO: Move to environment variable
const GITHUB_USERNAME = 'heoj1N'

export default async function Page() {
  // Fetch repositories and their README excerpts
  const repositories = await getRepositories(GITHUB_USERNAME)
  const featuredRepos = repositories.slice(0, 2) // Display only the top 2 or 3 repos
  
  const reposWithReadme = await Promise.all(
    featuredRepos.map(async (repo) => {
      const readmeExcerpt = await getReadmeContent(GITHUB_USERNAME, repo.name)
      return { ...repo, readmeExcerpt }
    })
  )

  return (
    <section>
      <div className="relative h-[300px] w-full mb-8">
        <div className="flex items-center justify-center absolute inset-0">
          <div className="relative w-full max-w-5xl mx-auto">
            {/* Left vortex - positioned absolutely */}
            <div className="absolute left-0 top-1/2 transform -translate-y-1/2">
              <VortexBlackHole width={300} height={300} color="#47a3f3" />
            </div>
            
            {/* Center title */}
            <h1 className="text-4xl font-semibold tracking-tighter text-white drop-shadow-lg z-10 text-center">
              Philipp's Portfolio
            </h1>
            
            {/* Right neural network - positioned absolutely */}
            <div className="absolute right-0 top-1/2 transform -translate-y-1/2">
              <NeuralNetwork width={300} height={300} />
            </div>
          </div>
        </div>
      </div>
      <p className="mb-4">
        {`
        I'm a Computer Science student at the Technical University of Munich,
        specialized in Artificial Intelligence, Robotics, and Biology.
        My interests in these fields are driven by my passion for understanding
        the underlying principles of intelligent systems and their applications.
        Among the subfields I'm interested in, are Natural Language Processing, Reinforcement
        Learning and Neuroscience.
        `}
      </p>
      
      {/* Research Papers */}
      <div className="my-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-semibold">Research Papers</h2>
          <Link href="/papers" className="text-sm text-blue-500 hover:underline">
            View all papers →
          </Link>
        </div>
        <Papers />
      </div>

      {/* Latest Blog Posts */}
      <div className="my-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-semibold">Latest Blog Posts</h2>
          <Link href="/blog" className="text-sm text-blue-500 hover:underline">
            View all posts →
          </Link>
        </div>
        <BlogPosts />
      </div>

      {/* Featured Projects */}
      <div className="my-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-semibold">Featured Projects</h2>
          <Link href="/projects" className="text-sm text-blue-500 hover:underline">
            View all projects →
          </Link>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          {reposWithReadme.map((repo) => (
            <RepositoryCard
              key={repo.name}
              repo={repo}
              readmeExcerpt={repo.readmeExcerpt}
            />
          ))}
        </div>
      </div>

      {/* Interactive Applications */}
      <div className="my-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-semibold">Interactive Applications</h2>
          <Link href="/apps" className="text-sm text-blue-500 hover:underline">
            View all apps →
          </Link>
        </div>
        <p>Check out my interactive demos and applications, including a Reinforcement Learning demo.</p>
      </div>
    </section>
  )
}
