import CogwheelLoader from './components/CogwheelLoader';

export default function Loading() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <CogwheelLoader size={60} color="#4F46E5" />
    </div>
  );
} 