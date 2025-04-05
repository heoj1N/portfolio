'use client'

import { useState } from 'react'

export type Comment = {
  id: string
  text: string
  author: string
  createdAt: string
}

type CommentsProps = {
  postSlug: string
  initialComments?: Comment[]
}

export default function Comments({ postSlug, initialComments = [] }: CommentsProps) {
  const [comments, setComments] = useState<Comment[]>(initialComments)
  const [newComment, setNewComment] = useState('')
  const [author, setAuthor] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!newComment.trim() || !author.trim()) {
      return
    }
    
    setIsSubmitting(true)
    
    // In a real implementation, this would send data to an API
    // For this demo, we'll just add it to the local state
    const comment: Comment = {
      id: Date.now().toString(),
      text: newComment.trim(),
      author: author.trim(),
      createdAt: new Date().toISOString()
    }
    
    setComments([...comments, comment])
    setNewComment('')
    setIsSubmitting(false)
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="mt-10 pt-10 border-t border-neutral-200 dark:border-neutral-700">
      <h2 className="text-xl font-semibold mb-4">Comments</h2>
      
      {comments.length > 0 ? (
        <div className="space-y-6 mb-8">
          {comments.map((comment) => (
            <div 
              key={comment.id} 
              className="bg-neutral-50 dark:bg-neutral-800 p-4 rounded-lg"
            >
              <div className="flex justify-between items-start">
                <p className="font-medium">{comment.author}</p>
                <span className="text-xs text-neutral-500 dark:text-neutral-400">
                  {formatDate(comment.createdAt)}
                </span>
              </div>
              <p className="mt-2 text-neutral-700 dark:text-neutral-300">{comment.text}</p>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-neutral-500 dark:text-neutral-400 mb-8">No comments yet. Be the first to comment!</p>
      )}
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="author" className="block text-sm font-medium mb-1">
            Name
          </label>
          <input
            type="text"
            id="author"
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-700 rounded-md bg-white dark:bg-neutral-900"
            required
          />
        </div>
        
        <div>
          <label htmlFor="comment" className="block text-sm font-medium mb-1">
            Comment
          </label>
          <textarea
            id="comment"
            rows={4}
            value={newComment}
            onChange={(e) => setNewComment(e.target.value)}
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-700 rounded-md bg-white dark:bg-neutral-900"
            required
          />
        </div>
        
        <button
          type="submit"
          disabled={isSubmitting}
          className="px-4 py-2 bg-neutral-800 dark:bg-neutral-200 text-white dark:text-black rounded-md hover:bg-neutral-700 dark:hover:bg-neutral-300 transition-colors disabled:opacity-50"
        >
          {isSubmitting ? 'Submitting...' : 'Submit Comment'}
        </button>
      </form>
    </div>
  )
} 