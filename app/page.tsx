import { BlogPosts } from 'app/components/posts'

export default function Page() {
  return (
    <section>
      <h1 className="mb-8 text-2xl font-semibold tracking-tighter">
        My Portfolio
      </h1>
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
      <div className="my-8">
        <BlogPosts />
      </div>
    </section>
  )
}
