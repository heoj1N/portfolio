'use client';

import { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface NeuralNetworkProps {
  width?: number;
  height?: number;
}

export default function NeuralNetwork({ width = 400, height = 300 }: NeuralNetworkProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const nodesRef = useRef<THREE.Mesh[]>([]);
  const linesRef = useRef<THREE.Line[]>([]);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Initialize camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;
    cameraRef.current = camera;

    // Initialize renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Create nodes
    const nodeGeometry = new THREE.SphereGeometry(0.1, 32, 32);
    const nodeMaterial = new THREE.MeshPhongMaterial({ color: 0x47a3f3 });
    const nodes: THREE.Mesh[] = [];

    // Create a 3x3 grid of nodes
    for (let i = -1; i <= 1; i++) {
      for (let j = -1; j <= 1; j++) {
        for (let k = -1; k <= 1; k++) {
          const node = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
          node.position.set(i, j, k);
          scene.add(node);
          nodes.push(node);
        }
      }
    }
    nodesRef.current = nodes;

    // Create connections
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0x47a3f3, opacity: 0.3, transparent: true });
    const lines: THREE.Line[] = [];

    // Connect nodes to their neighbors
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const distance = nodes[i].position.distanceTo(nodes[j].position);
        if (distance < 1.5) { // Only connect nearby nodes
          const geometry = new THREE.BufferGeometry().setFromPoints([
            nodes[i].position,
            nodes[j].position,
          ]);
          const line = new THREE.Line(geometry, lineMaterial.clone());
          scene.add(line);
          lines.push(line);
        }
      }
    }
    linesRef.current = lines;

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Animation
    let time = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      time += 0.01;

      // Rotate the entire scene
      scene.rotation.y = time * 0.2;

      // Animate nodes
      nodes.forEach((node, index) => {
        const material = node.material as THREE.MeshPhongMaterial;
        material.color.setHSL(
          (Math.sin(time + index * 0.1) + 1) / 2,
          0.7,
          0.5
        );
      });

      // Animate lines
      lines.forEach((line, index) => {
        const material = line.material as THREE.LineBasicMaterial;
        material.opacity = 0.3 + Math.sin(time + index * 0.05) * 0.1;
      });

      renderer.render(scene, camera);
    };

    animate();

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current || !camera || !renderer) return;
      
      const newWidth = containerRef.current.clientWidth;
      const newHeight = containerRef.current.clientHeight;
      
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      containerRef.current?.removeChild(renderer.domElement);
      scene.remove(...nodes, ...lines);
      renderer.dispose();
    };
  }, [width, height]);

  return <div ref={containerRef} className="w-full h-full" />;
} 