import { getBlogPosts } from 'app/blog/utils'
import { getPapers } from 'app/papers/utils'

export const baseUrl = 'https://portfolio-blog-starter.vercel.app'

export default async function sitemap() {
  let blogs = getBlogPosts().map((post) => ({
    url: `${baseUrl}/blog/${post.slug}`,
    lastModified: post.metadata.publishedAt,
  }))

  let papers = getPapers().map((paper) => ({
    url: `${baseUrl}/papers/${paper.slug}`,
    lastModified: paper.metadata.publishedAt,
  }))

  let routes = ['', '/apps', '/blog', '/papers', '/projects'].map((route) => ({
    url: `${baseUrl}${route}`,
    lastModified: new Date().toISOString().split('T')[0],
  }))

  return [...routes, ...blogs, ...papers]
}
