'use client';

import { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface VortexBlackHoleProps {
  width?: number;
  height?: number;
  ringsCount?: number;
  color?: string;
}

export default function VortexBlackHole({ 
  width = 400, 
  height = 300, 
  ringsCount = 7,
  color = '#47a3f3'
}: VortexBlackHoleProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const ringsRef = useRef<THREE.Mesh[]>([]);
  const groupRef = useRef<THREE.Group | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Initialize camera with wider field of view and farther position
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.z = 4.5; // Move camera farther to see whole donut
    cameraRef.current = camera;

    // Initialize renderer with transparency
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true, 
      alpha: true,
      powerPreference: 'high-performance' 
    });
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0); // Transparent background
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Create a group to hold all the rings
    const group = new THREE.Group();
    scene.add(group);
    groupRef.current = group;

    // Create the vortex rings
    const rings: THREE.Mesh[] = [];
    const baseColor = new THREE.Color(color);
    
    // Use a smaller scale factor to make the donut more visible
    const scaleFactor = 0.75;
    
    for (let i = 0; i < ringsCount; i++) {
      const ringRadius = (1.2 - i * 0.1) * scaleFactor;
      const tubeRadius = (0.03 + i * 0.01) * scaleFactor;
      
      const torusGeometry = new THREE.TorusGeometry(
        ringRadius, // radius
        tubeRadius, // tube radius
        24, // radial segments - increased for smoother appearance
        96 // tubular segments - increased for smoother appearance
      );
      
      // Create a gradient-colored material
      const intensity = 0.8 - (i / ringsCount) * 0.6;
      const ringColor = baseColor.clone().multiplyScalar(intensity);
      
      const material = new THREE.MeshPhongMaterial({
        color: ringColor,
        shininess: 80,
        transparent: true,
        opacity: 0.9 - (i / ringsCount) * 0.3,
        side: THREE.DoubleSide
      });
      
      const ring = new THREE.Mesh(torusGeometry, material);
      
      // Rotate each ring to create the spiral effect
      ring.rotation.x = Math.PI / 2; // Make rings horizontal
      ring.rotation.z = (i / ringsCount) * Math.PI * 0.5; // Progressive rotation for spiral
      
      group.add(ring);
      rings.push(ring);
    }
    
    // Add a small sphere in the center for the black hole "center"
    const coreGeometry = new THREE.SphereGeometry(0.15 * scaleFactor, 32, 32);
    const coreMaterial = new THREE.MeshBasicMaterial({ 
      color: 0x000000,
      transparent: true,
      opacity: 0.9
    });
    const core = new THREE.Mesh(coreGeometry, coreMaterial);
    group.add(core);
    
    ringsRef.current = rings;

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.9);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    const backLight = new THREE.DirectionalLight(0xffffff, 0.5);
    backLight.position.set(-1, -1, -1);
    scene.add(backLight);

    // Animation
    let time = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      time += 0.01;

      if (group) {
        // Rotate mostly on z-axis for best visibility of rings
        group.rotation.z += 0.005;
        // Very minimal y-rotation to keep donut face visible
        group.rotation.y = Math.sin(time * 0.2) * 0.1;
        // Small x-rotation to show more of the donut face
        group.rotation.x = Math.PI * 0.05;
        
        // Pulse the rings
        rings.forEach((ring, index) => {
          const material = ring.material as THREE.MeshPhongMaterial;
          const pulseIntensity = 0.1 * Math.sin(time * 2 + index * 0.3);
          
          // Scale rings for pulsing effect
          ring.scale.set(
            1 + pulseIntensity, 
            1 + pulseIntensity, 
            1 + pulseIntensity
          );
          
          // Adjust opacity for flowing effect
          material.opacity = 0.7 + 0.3 * Math.sin(time * 1.5 + index * 0.4);
        });
      }

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
      renderer.dispose();
    };
  }, [width, height, ringsCount, color]);

  return <div ref={containerRef} className="w-full h-full" />;
} 