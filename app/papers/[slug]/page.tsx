import { notFound } from 'next/navigation'
import { CustomMDX } from 'app/components/mdx'
import { formatDate, getPapers } from 'app/papers/utils'
import { baseUrl } from 'app/sitemap'
import { PaperChat } from '../../components/PaperChat'

export async function generateStaticParams() {
  let papers = getPapers()

  return papers.map((paper) => ({
    slug: paper.slug,
  }))
}

export function generateMetadata({ params }) {
  let paper = getPapers().find((paper) => paper.slug === params.slug)
  if (!paper) {
    return
  }

  let {
    title,
    publishedAt: publishedTime,
    summary: description,
    image,
  } = paper.metadata
  let ogImage = image
    ? image
    : `${baseUrl}/og?title=${encodeURIComponent(title)}`

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      type: 'article',
      publishedTime,
      url: `${baseUrl}/papers/${paper.slug}`,
      images: [
        {
          url: ogImage,
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      images: [ogImage],
    },
  }
}

export default function Paper({ params }) {
  let paper = getPapers().find((paper) => paper.slug === params.slug)

  if (!paper) {
    notFound()
  }

  return (
    <section>
      <script
        type="application/ld+json"
        suppressHydrationWarning
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'ScholarlyArticle',
            headline: paper.metadata.title,
            datePublished: paper.metadata.publishedAt,
            author: paper.metadata.authors ? 
              paper.metadata.authors.split(',').map(name => ({
                '@type': 'Person',
                'name': name.trim()
              })) : 
              {
                '@type': 'Person',
                name: 'Unnamed Author',
              },
            publisher: {
              '@type': 'Organization',
              'name': paper.metadata.journal || 'Unknown Journal'
            },
            description: paper.metadata.summary,
            url: `${baseUrl}/papers/${paper.slug}`,
          }),
        }}
      />
      <h1 className="title font-semibold text-2xl tracking-tighter">
        {paper.metadata.title}
      </h1>
      <div className="flex flex-col mt-2 mb-8 text-sm">
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          {formatDate(paper.metadata.publishedAt)}
        </p>
        {paper.metadata.authors && (
          <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
            Authors: {paper.metadata.authors}
          </p>
        )}
        {paper.metadata.journal && (
          <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
            Journal: {paper.metadata.journal}
          </p>
        )}
        {paper.metadata.doi && (
          <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
            DOI: <a href={`https://doi.org/${paper.metadata.doi}`} className="underline">{paper.metadata.doi}</a>
          </p>
        )}
      </div>
      <article className="prose">
        <CustomMDX source={paper.content} />
      </article>
      
      {/* Paper Chat Component */}
      <PaperChat paperSlug={paper.slug} paperTitle={paper.metadata.title} />
    </section>
  )
} 