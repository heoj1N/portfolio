import ReinforcementLearningDemo from 'app/components/ReinforcementLearningDemo'
import Link from 'next/link'

export default function AppsPage() {
  return (
    <section>
      <h1 className="font-semibold text-3xl mb-6 tracking-tighter">
        Interactive Applications
      </h1>
      
      <div className="grid gap-6 grid-cols-1 md:grid-cols-2 mb-8">
        <Link href="/apps/visuals" className="block p-4 border border-gray-700 rounded-lg hover:bg-gray-800 transition-colors">
          <h3 className="text-xl font-medium mb-2">Neural Network & Vortex Visualizations</h3>
          <p className="text-gray-400">
            Explore interactive 3D visualizations of neural networks and vortex structures with customizable parameters.
          </p>
        </Link>
        
        {/* Add more application cards here as needed */}
      </div>
      
      <div className="my-8">
        <h2 className="text-2xl font-semibold mb-4">Reinforcement Learning Trainer</h2>
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