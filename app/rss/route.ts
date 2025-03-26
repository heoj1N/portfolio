import { baseUrl } from 'app/sitemap'
import { getBlogPosts } from 'app/blog/utils'
import { getPapers } from 'app/papers/utils'

export async function GET() {
  let allBlogs = await getBlogPosts()
  let allPapers = await getPapers()

  // Combine blogs and papers into a single feed
  const blogItems = allBlogs.map(post => ({
    title: post.metadata.title,
    link: `${baseUrl}/blog/${post.slug}`,
    description: post.metadata.summary || '',
    pubDate: new Date(post.metadata.publishedAt).toUTCString(),
    type: 'blog'
  }))

  const paperItems = allPapers.map(paper => ({
    title: paper.metadata.title,
    link: `${baseUrl}/papers/${paper.slug}`,
    description: paper.metadata.summary || '',
    pubDate: new Date(paper.metadata.publishedAt).toUTCString(),
    type: 'paper'
  }))

  const allItems = [...blogItems, ...paperItems].sort((a, b) => {
    if (new Date(a.pubDate) > new Date(b.pubDate)) {
      return -1
    }
    return 1
  })

  const itemsXml = allItems
    .map(
      (item) =>
        `<item>
          <title>${item.title}${item.type === 'paper' ? ' (Paper)' : ''}</title>
          <link>${item.link}</link>
          <description>${item.description}</description>
          <pubDate>${item.pubDate}</pubDate>
        </item>`
    )
    .join('\n')

  const rssFeed = `<?xml version="1.0" encoding="UTF-8" ?>
  <rss version="2.0">
    <channel>
        <title>My Portfolio</title>
        <link>${baseUrl}</link>
        <description>This is my portfolio RSS feed with blog posts and papers</description>
        ${itemsXml}
    </channel>
  </rss>`

  return new Response(rssFeed, {
    headers: {
      'Content-Type': 'text/xml',
    },
  })
}
