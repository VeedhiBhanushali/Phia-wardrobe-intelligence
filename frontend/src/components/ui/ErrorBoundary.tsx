"use client";

import { Component, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      return (
        <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
          <AlertTriangle size={32} className="text-phia-gray-300 mb-3" />
          <p className="text-sm font-medium text-phia-black mb-1">
            Something went wrong
          </p>
          <p className="text-xs text-phia-gray-400 mb-4 max-w-[260px]">
            {this.state.error?.message || "An unexpected error occurred"}
          </p>
          <button
            onClick={this.handleRetry}
            className="flex items-center gap-1.5 px-4 py-2 rounded-full bg-phia-black text-white text-sm"
          >
            <RefreshCw size={14} />
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export function InlineError({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="rounded-2xl bg-phia-gray-50 p-5 text-center">
      <p className="text-sm text-phia-gray-400 mb-1">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="text-xs text-phia-black font-medium mt-2 underline underline-offset-2"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export default ErrorBoundary;
