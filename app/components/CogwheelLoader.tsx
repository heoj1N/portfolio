'use client';

import { motion } from 'framer-motion';

interface CogwheelLoaderProps {
  size?: number;
  color?: string;
}

export default function CogwheelLoader({ size = 40, color = '#000000' }: CogwheelLoaderProps) {
  return (
    <div className="flex items-center justify-center">
      <motion.svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <motion.path
          d="M12 2L14 6L18 7L14 8L12 12L10 8L6 7L10 6L12 2Z"
          fill={color}
          animate={{
            rotate: 360,
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "linear",
          }}
        />
        <motion.path
          d="M12 22L14 18L18 17L14 16L12 12L10 16L6 17L10 18L12 22Z"
          fill={color}
          animate={{
            rotate: -360,
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      </motion.svg>
    </div>
  );
} 