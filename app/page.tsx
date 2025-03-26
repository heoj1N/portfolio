import { BlogPosts } from 'app/components/posts'
import NeuralNetwork from 'app/components/NeuralNetwork'
import ReinforcementLearningDemo from 'app/components/ReinforcementLearningDemo'

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
        <h2 className="text-2xl font-semibold mb-4">Latest Blog Posts</h2>
        <BlogPosts />
      </div>
      <div className="my-8">
        <h2 className="text-2xl font-semibold mb-4">Reinforcement Learning Demo</h2>
        <p className="mb-4">
          This interactive demo showcases Q-learning, a fundamental reinforcement learning algorithm. 
          The agent (🤖) learns to navigate through a grid environment to reach the target (🎯) 
          while avoiding obstacles (🟫). The brightness of each cell represents the learned value 
          of that state (brighter = higher value).
        </p>
        <p className="mb-6">
          <strong>Controls:</strong> Adjust the learning rate and discount factor to see how they 
          affect the learning process. The learning rate controls how quickly the agent incorporates 
          new information, while the discount factor determines how much the agent values future rewards 
          compared to immediate ones.
        </p>
        <ReinforcementLearningDemo />
      </div>
    </section>
  )
}
