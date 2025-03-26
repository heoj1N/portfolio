'use client';

import { useEffect, useRef, useState } from 'react';
import styles from './ReinforcementLearningDemo.module.css';

interface Position {
  x: number;
  y: number;
}

interface QTable {
  [key: string]: number[];
}

interface State {
  agent: Position;
  target: Position;
  obstacles: Position[];
  episode: number;
  steps: number;
  isLearning: boolean;
  learningRate: number;
  discountFactor: number;
}

export default function ReinforcementLearningDemo() {
  const [state, setState] = useState<State>({
    agent: { x: 0, y: 0 },
    target: { x: 4, y: 4 },
    obstacles: [
      { x: 2, y: 2 },
      { x: 3, y: 2 },
      { x: 2, y: 3 },
    ],
    episode: 0,
    steps: 0,
    isLearning: false,
    learningRate: 0.1,
    discountFactor: 0.95,
  });

  const qTableRef = useRef<QTable>({});
  const animationFrameRef = useRef<number>();
  const isLearningRef = useRef(false);

  // Initialize Q-table
  useEffect(() => {
    const actions = ['up', 'right', 'down', 'left'];
    for (let x = 0; x < 5; x++) {
      for (let y = 0; y < 5; y++) {
        qTableRef.current[`${x},${y}`] = actions.map(() => 0);
      }
    }
  }, []);

  // Q-learning algorithm
  const updateQValue = (currentState: string, action: number, reward: number, nextState: string) => {
    const currentQ = qTableRef.current[currentState][action];
    const nextMaxQ = Math.max(...qTableRef.current[nextState]);
    
    const newQ = currentQ + state.learningRate * (reward + state.discountFactor * nextMaxQ - currentQ);
    qTableRef.current[currentState][action] = newQ;
  };

  // Get next action using epsilon-greedy policy
  const getNextAction = (currentState: string, episodeCount: number): number => {
    const epsilon = Math.max(0.1, 1 - episodeCount / 1000);
    if (Math.random() < epsilon) {
      return Math.floor(Math.random() * 4);
    }
    return qTableRef.current[currentState].indexOf(Math.max(...qTableRef.current[currentState]));
  };

  // Check if position is valid
  const isValidPosition = (pos: Position, obstacles: Position[]): boolean => {
    return (
      pos.x >= 0 &&
      pos.x < 5 &&
      pos.y >= 0 &&
      pos.y < 5 &&
      !obstacles.some(obs => obs.x === pos.x && obs.y === pos.y)
    );
  };

  // Move agent based on action
  const moveAgent = (action: number, currentPos: Position, obstacles: Position[]): Position => {
    const newPos = { ...currentPos };
    switch (action) {
      case 0: // up
        newPos.y--;
        break;
      case 1: // right
        newPos.x++;
        break;
      case 2: // down
        newPos.y++;
        break;
      case 3: // left
        newPos.x--;
        break;
    }
    return isValidPosition(newPos, obstacles) ? newPos : currentPos;
  };

  // Calculate reward
  const calculateReward = (pos: Position, target: Position): number => {
    if (pos.x === target.x && pos.y === target.y) {
      return 100;
    }
    return -1;
  };

  // Learning loop
  const learn = () => {
    if (!isLearningRef.current) {
      return;
    }

    setState(prevState => {
      // Use the current state values to calculate the next state
      const currentState = `${prevState.agent.x},${prevState.agent.y}`;
      const action = getNextAction(currentState, prevState.episode);
      const newPos = moveAgent(action, prevState.agent, prevState.obstacles);
      const nextState = `${newPos.x},${newPos.y}`;
      const reward = calculateReward(newPos, prevState.target);

      updateQValue(currentState, action, reward, nextState);

      // Check if agent reached the target
      if (newPos.x === prevState.target.x && newPos.y === prevState.target.y) {
        return {
          ...prevState,
          agent: { x: 0, y: 0 }, // Reset agent position
          episode: prevState.episode + 1,
          steps: 0,
        };
      }

      return {
        ...prevState,
        agent: newPos,
        steps: prevState.steps + 1,
      };
    });

    // Schedule the next animation frame
    animationFrameRef.current = requestAnimationFrame(learn);
  };

  // Start/stop learning
  const toggleLearning = () => {
    setState(prev => {
      const newIsLearning = !prev.isLearning;
      isLearningRef.current = newIsLearning;
      
      return { ...prev, isLearning: newIsLearning };
    });

    if (!state.isLearning) {
      // If we're starting learning, kick off the animation frame
      cancelAnimationFrame(animationFrameRef.current!);
      animationFrameRef.current = requestAnimationFrame(learn);
    }
  };

  // Reset environment
  const resetEnvironment = () => {
    isLearningRef.current = false;
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    setState({
      agent: { x: 0, y: 0 },
      target: { x: 4, y: 4 },
      obstacles: [
        { x: 2, y: 2 },
        { x: 3, y: 2 },
        { x: 2, y: 3 },
      ],
      episode: 0,
      steps: 0,
      isLearning: false,
      learningRate: state.learningRate,
      discountFactor: state.discountFactor,
    });
    
    // Reset Q-table
    const actions = ['up', 'right', 'down', 'left'];
    for (let x = 0; x < 5; x++) {
      for (let y = 0; y < 5; y++) {
        qTableRef.current[`${x},${y}`] = actions.map(() => 0);
      }
    }
  };

  // Cleanup
  useEffect(() => {
    return () => {
      isLearningRef.current = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return (
    <div className={styles.container}>
      <div className={styles.grid}>
        {Array.from({ length: 5 }, (_, y) =>
          Array.from({ length: 5 }, (_, x) => {
            const isAgent = x === state.agent.x && y === state.agent.y;
            const isTarget = x === state.target.x && y === state.target.y;
            const isObstacle = state.obstacles.some(obs => obs.x === x && obs.y === y);
            const qValues = qTableRef.current[`${x},${y}`] || [0, 0, 0, 0];
            const maxQ = Math.max(...qValues);

            return (
              <div
                key={`${x},${y}`}
                className={`${styles.cell} ${
                  isAgent ? styles.agent :
                  isTarget ? styles.target :
                  isObstacle ? styles.obstacle :
                  styles.empty
                }`}
                style={{
                  opacity: isAgent || isTarget || isObstacle ? 1 : 0.3 + maxQ / 100,
                }}
              >
                {isAgent && '🤖'}
                {isTarget && '🎯'}
                {isObstacle && '🟫'}
              </div>
            );
          })
        )}
      </div>
      <div className={styles.controls}>
        <button
          onClick={toggleLearning}
          className={styles.button}
        >
          {state.isLearning ? 'Stop Learning' : 'Start Learning'}
        </button>
        <button
          onClick={resetEnvironment}
          className={styles.button}
        >
          Reset
        </button>
        <div className={styles.stats}>
          <p>Episode: {state.episode}</p>
          <p>Steps: {state.steps}</p>
        </div>
        <div className={styles.parameters}>
          <label>
            Learning Rate:
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={state.learningRate}
              onChange={(e) => setState(prev => ({
                ...prev,
                learningRate: parseFloat(e.target.value)
              }))}
            />
            {state.learningRate.toFixed(2)}
          </label>
          <label>
            Discount Factor:
            <input
              type="range"
              min="0.1"
              max="0.99"
              step="0.01"
              value={state.discountFactor}
              onChange={(e) => setState(prev => ({
                ...prev,
                discountFactor: parseFloat(e.target.value)
              }))}
            />
            {state.discountFactor.toFixed(2)}
          </label>
        </div>
      </div>
    </div>
  );
} 