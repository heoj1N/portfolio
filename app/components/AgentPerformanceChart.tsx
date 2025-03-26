'use client';

import React from 'react';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  Title, 
  Tooltip, 
  Legend 
} from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface PerformanceDataPoint {
  episode: number;
  numMoves?: number;
  avgMoves?: number;
  wasSuccessful?: number;
  successRate?: number;
}

interface SummaryStats {
  totalEpisodes: number;
  successRate: number;
  avgMoves: number;
}

interface PerformanceData {
  performanceOverTime: PerformanceDataPoint[];
  summaryStats: SummaryStats;
}

interface AgentPerformanceChartProps {
  performanceData: PerformanceData;
}

const AgentPerformanceChart: React.FC<AgentPerformanceChartProps> = ({ performanceData }) => {
  const { performanceOverTime, summaryStats } = performanceData;
  
  if (!performanceOverTime || performanceOverTime.length === 0) {
    return <div className="text-center p-4">Not enough data to display performance metrics</div>;
  }
  
  // Prepare data for Chart.js
  const chartData = {
    labels: performanceOverTime.map(data => `Episode ${data.episode}`),
    datasets: [
      {
        label: 'Avg. Moves',
        data: performanceOverTime.map(data => data.avgMoves || data.numMoves),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      },
      {
        label: 'Success Rate',
        data: performanceOverTime.map(data => 
          data.successRate !== undefined ? data.successRate : (data.wasSuccessful ? 1 : 0)
        ),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        yAxisID: 'y1',
      },
    ],
  };
  
  const options = {
    responsive: true,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    scales: {
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
        title: {
          display: true,
          text: 'Number of Moves'
        }
      },
      y1: {
        type: 'linear' as const,
        display: true,
        position: 'right' as const,
        title: {
          display: true,
          text: 'Success Rate'
        },
        min: 0,
        max: 1,
        grid: {
          drawOnChartArea: false,
        },
      },
    },
  };

  return (
    <div className="performance-metrics p-4 bg-black bg-opacity-80 rounded-lg border border-cyan-500/20 shadow">
      <h3 className="text-xl font-semibold mb-4 text-cyan-400">Agent Performance Metrics</h3>
      
      <div className="summary-stats grid grid-cols-3 gap-4 mb-6">
        <div className="stat-item p-3 bg-black bg-opacity-30 rounded border border-cyan-500/20">
          <div className="text-sm text-cyan-400/70">Total Episodes</div>
          <div className="text-2xl font-bold text-cyan-400">{summaryStats.totalEpisodes}</div>
        </div>
        <div className="stat-item p-3 bg-black bg-opacity-30 rounded border border-cyan-500/20">
          <div className="text-sm text-cyan-400/70">Success Rate</div>
          <div className="text-2xl font-bold text-cyan-400">{(summaryStats.successRate * 100).toFixed(1)}%</div>
        </div>
        <div className="stat-item p-3 bg-black bg-opacity-30 rounded border border-cyan-500/20">
          <div className="text-sm text-cyan-400/70">Avg. Moves per Episode</div>
          <div className="text-2xl font-bold text-cyan-400">{summaryStats.avgMoves.toFixed(1)}</div>
        </div>
      </div>
      
      <div className="performance-chart">
        <h4 className="text-lg font-medium mb-2 text-cyan-400/90">Learning Curve</h4>
        <div className="h-64">
          <Line options={options} data={chartData} />
        </div>
      </div>
    </div>
  );
};

export default AgentPerformanceChart; 