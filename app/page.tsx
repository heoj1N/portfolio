import { BlogPosts } from 'app/components/posts'
import NeuralNetwork from 'app/components/NeuralNetwork'

export default function Page() {
  return (
    <section>
      <div className="relative h-[400px] w-full mb-8">
        <NeuralNetwork width={800} height={400} />
        <div className="absolute inset-0 flex items-center justify-center">
          <h1 className="text-4xl font-semibold tracking-tighter text-white drop-shadow-lg">
            My Portfolio
          </h1>
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
      <div className="my-8">
        <BlogPosts />
      </div>
    </section>
  )
}
