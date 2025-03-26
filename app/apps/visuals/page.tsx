'use client';

import React, { useState } from 'react';
import NeuralNetwork from 'app/components/NeuralNetwork';
import VortexBlackHole from 'app/components/VortexBlackHole';

export default function VisualsPage() {
  const [vortexColor, setVortexColor] = useState('#47a3f3');
  const [vortexRings, setVortexRings] = useState(7);
  const [componentSize, setComponentSize] = useState(400);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Neural Network & Vortex Visualizations</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {/* VortexBlackHole Section */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-2xl font-semibold mb-4">Vortex Black Hole</h2>
          <div className="h-[450px] w-full flex items-center justify-center mb-4">
            <VortexBlackHole 
              width={componentSize} 
              height={componentSize} 
              color={vortexColor} 
              ringsCount={vortexRings} 
            />
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Color</label>
              <div className="flex gap-2">
                <input 
                  type="color" 
                  value={vortexColor}
                  onChange={(e) => setVortexColor(e.target.value)}
                  className="rounded"
                />
                <span>{vortexColor}</span>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Number of Rings: {vortexRings}</label>
              <input 
                type="range" 
                min="3" 
                max="12" 
                value={vortexRings}
                onChange={(e) => setVortexRings(Number(e.target.value))}
                className="w-full"
              />
            </div>
            
            <p className="text-sm text-gray-400 mt-4">
              This donut-shaped vortex visualization resembles a black hole with a spinning effect. 
              The camera position has been adjusted to show the entire structure from the optimal angle.
            </p>
          </div>
        </div>
        
        {/* NeuralNetwork Section */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-2xl font-semibold mb-4">Neural Network</h2>
          <div className="h-[450px] w-full flex items-center justify-center mb-4">
            <NeuralNetwork width={componentSize} height={componentSize} />
          </div>
          
          <div>
            <p className="text-sm text-gray-400 mb-4">
              This visualization demonstrates a neural network structure with interconnected nodes.
              The nodes change color and connections pulse to represent data flow through the network.
            </p>
          </div>
        </div>
      </div>
      
      {/* Common Controls */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Common Settings</h2>
        
        <div>
          <label className="block text-sm font-medium mb-1">Visualization Size: {componentSize}px</label>
          <input 
            type="range" 
            min="300" 
            max="800" 
            step="50"
            value={componentSize}
            onChange={(e) => setComponentSize(Number(e.target.value))}
            className="w-full"
          />
        </div>
      </div>
      
      <div className="mt-8">
        <h2 className="text-2xl font-semibold mb-4">About These Visualizations</h2>
        <p className="mb-4">
          These 3D visualizations are created using Three.js. The vortex black hole 
          represents a spinning donut-shaped vortex with concentric rings, while the neural network shows 
          interconnected nodes representing a simplified neural network structure.
        </p>
        <p>
          The vortex visualization has been optimized to ensure the entire donut shape is visible 
          at all times during rotation. Both components are positioned symmetrically on the homepage 
          to create a balanced design around the title.
        </p>
      </div>
    </div>
  );
} 