// Performance tracking for reinforcement learning agent

interface EpisodeStats {
  problemId: string;
  numMoves: number;
  wasSuccessful: boolean;
  timestamp: number;
}

interface SummaryStats {
  totalEpisodes: number;
  successRate: number;
  avgMoves: number;
}

interface PerformanceDataPoint {
  episode: number;
  numMoves?: number;
  avgMoves?: number;
  wasSuccessful?: number;
  successRate?: number;
}

class AgentPerformanceTracker {
  private episodeStats: EpisodeStats[];
  private optimalSolutions: Record<string, number>; // Map problem IDs to their optimal solutions

  constructor() {
    this.episodeStats = [];
    this.optimalSolutions = {}; 
  }

  // Record performance for a completed episode
  recordEpisode(problemId: string, numMoves: number, wasSuccessful: boolean): void {
    this.episodeStats.push({
      problemId,
      numMoves,
      wasSuccessful,
      timestamp: Date.now(),
    });
  }

  // Set the optimal solution for a specific problem
  setOptimalSolution(problemId: string, optimalMoves: number): void {
    this.optimalSolutions[problemId] = optimalMoves;
  }

  // Calculate efficiency ratio (actual/optimal)
  getEfficiencyRatio(problemId: string, numMoves: number): number | null {
    const optimal = this.optimalSolutions[problemId];
    if (!optimal) return null;
    return optimal / numMoves; // Higher is better (optimal=1.0)
  }

  // Get performance metrics over time
  getPerformanceOverTime(problemId: string | null = null, windowSize: number = 10): PerformanceDataPoint[] {
    // Filter by problem ID if specified
    const relevantStats = problemId 
      ? this.episodeStats.filter(stat => stat.problemId === problemId)
      : this.episodeStats;
      
    // If we don't have enough data for a rolling window, return raw data
    if (relevantStats.length < windowSize) {
      return relevantStats.map((stat, i) => ({
        episode: i + 1,
        numMoves: stat.numMoves,
        wasSuccessful: stat.wasSuccessful ? 1 : 0
      }));
    }
    
    // Calculate rolling average of moves
    const rollingAverages: PerformanceDataPoint[] = [];
    for (let i = windowSize; i <= relevantStats.length; i++) {
      const window = relevantStats.slice(i - windowSize, i);
      const avgMoves = window.reduce((sum, stat) => sum + stat.numMoves, 0) / windowSize;
      rollingAverages.push({
        episode: i,
        avgMoves,
        successRate: window.filter(stat => stat.wasSuccessful).length / windowSize
      });
    }
    
    return rollingAverages;
  }
  
  // Get summary statistics
  getSummaryStats(): SummaryStats {
    if (this.episodeStats.length === 0) {
      return {
        totalEpisodes: 0,
        successRate: 0,
        avgMoves: 0
      };
    }
    
    return {
      totalEpisodes: this.episodeStats.length,
      successRate: this.episodeStats.filter(stat => stat.wasSuccessful).length / this.episodeStats.length,
      avgMoves: this.episodeStats.reduce((sum, stat) => sum + stat.numMoves, 0) / this.episodeStats.length
    };
  }
}

export default AgentPerformanceTracker; 