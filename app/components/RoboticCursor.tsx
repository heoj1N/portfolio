'use client';

import { useEffect, useState } from 'react';
import styles from './RoboticCursor.module.css';

// TODO: Add sound effects toggle functionality.
// TODO: Optimize performance for the grid effect.
// TODO: Add more robotic components to the trail.

interface Point {
  x: number;
  y: number;
}

interface GridCell {
  x: number;
  y: number;
  isHighlighted: boolean;
}

export default function RoboticCursor() {
  const [mousePosition, setMousePosition] = useState<Point>({ x: 0, y: 0 });
  const [trail, setTrail] = useState<Point[]>([]);
  const [gridCells, setGridCells] = useState<GridCell[]>([]);
  const [isHoveringLink, setIsHoveringLink] = useState(false);
  const CELL_SIZE = 50; // Match the CSS grid size
  const HIGHLIGHT_RADIUS = 100; // Radius for grid cell highlighting

  useEffect(() => {
    // Add cursor hiding class to body
    document.body.classList.add(styles.cursorHidden);
    return () => document.body.classList.remove(styles.cursorHidden);
  }, []);

  useEffect(() => {
    // Track whether we're hovering over a link
    const handleMouseOver = (e: MouseEvent) => {
      // Check if the target or any of its parents is a link
      let element = e.target as HTMLElement;
      let isLink = false;
      
      while (element && !isLink) {
        if (element.tagName === 'A' || element.tagName === 'BUTTON' || 
            element.getAttribute('role') === 'button') {
          isLink = true;
        }
        element = element.parentElement as HTMLElement;
      }
      
      setIsHoveringLink(isLink);
    };

    document.addEventListener('mouseover', handleMouseOver);
    return () => {
      document.removeEventListener('mouseover', handleMouseOver);
    };
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      // Get both client coordinates and scroll position
      const mouseX = e.clientX;
      const mouseY = e.clientY;
      const scrollX = window.scrollX;
      const scrollY = window.scrollY;
      
      // Update cursor and trail with client coordinates (viewport-relative)
      setMousePosition({ x: mouseX, y: mouseY });
      setTrail(prev => {
        const newTrail = [...prev, { x: mouseX, y: mouseY }];
        return newTrail.slice(-10);
      });

      // Calculate absolute position (document-relative)
      const absoluteX = mouseX + scrollX;
      const absoluteY = mouseY + scrollY;

      // Calculate grid cells to highlight
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const newGridCells: GridCell[] = [];

      // Calculate the range of cells to check (optimization)
      const startX = Math.max(0, Math.floor((absoluteX - HIGHLIGHT_RADIUS) / CELL_SIZE));
      const endX = Math.min(Math.ceil((scrollX + viewportWidth) / CELL_SIZE), Math.ceil((absoluteX + HIGHLIGHT_RADIUS) / CELL_SIZE));
      const startY = Math.max(0, Math.floor((absoluteY - HIGHLIGHT_RADIUS) / CELL_SIZE));
      const endY = Math.min(Math.ceil((scrollY + viewportHeight) / CELL_SIZE), Math.ceil((absoluteY + HIGHLIGHT_RADIUS) / CELL_SIZE));

      for (let x = startX; x < endX; x++) {
        for (let y = startY; y < endY; y++) {
          const cellCenterX = x * CELL_SIZE + CELL_SIZE / 2;
          const cellCenterY = y * CELL_SIZE + CELL_SIZE / 2;
          
          const distance = Math.sqrt(
            Math.pow(absoluteX - cellCenterX, 2) + 
            Math.pow(absoluteY - cellCenterY, 2)
          );

          if (distance <= HIGHLIGHT_RADIUS) {
            // Convert back to viewport-relative position for rendering
            newGridCells.push({
              x: x * CELL_SIZE - scrollX,
              y: y * CELL_SIZE - scrollY,
              isHighlighted: true
            });
          }
        }
      }

      setGridCells(newGridCells);
    };

    // Also handle scroll events to update grid cell positions
    const handleScroll = () => {
      // Trigger a mousemove event to update positions
      const lastMouseEvent = new MouseEvent('mousemove', {
        clientX: mousePosition.x,
        clientY: mousePosition.y
      });
      handleMouseMove(lastMouseEvent);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('scroll', handleScroll);
    };
  }, [mousePosition.x, mousePosition.y]);

  return (
    <>
      <div 
        className={`${styles.cursor} ${isHoveringLink ? styles.link : ''}`} 
        style={{ left: mousePosition.x, top: mousePosition.y }} 
      />
      <div className={styles.trail}>
        {trail.map((point, index) => (
          <div
            key={index}
            className={`${styles.trailDot} ${isHoveringLink ? styles.link : ''}`}
            style={{
              left: point.x,
              top: point.y,
              opacity: (index + 1) / trail.length,
            }}
          />
        ))}
      </div>
      <div className={styles.grid} />
      {gridCells.map((cell, index) => (
        <div
          key={index}
          className={`${styles.gridCell} ${styles.highlight} ${isHoveringLink ? styles.link : ''}`}
          style={{
            left: cell.x,
            top: cell.y,
          }}
        />
      ))}
    </>
  );
} 