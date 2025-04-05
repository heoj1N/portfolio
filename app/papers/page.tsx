import { Papers } from 'app/components/papers'

export const metadata = {
  title: 'Papers',
  description: 'Academic papers and research publications.',
}

export default function Page() {
  return (
    <section>
      <h1 className="font-semibold text-2xl mb-8 tracking-tighter">Research Papers</h1>
      <p className="mb-6">
        A collection of academic papers and research publications covering topics in artificial intelligence, 
        reinforcement learning, natural language processing, and related fields.
      </p>
      <Papers />
    </section>
  )
} 