import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { Toaster } from "sonner";

import "./styles/globals.css";
import { router } from "./app/router";
import { TooltipProvider } from "@/components/ui/Tooltip";

const qc = new QueryClient({
  defaultOptions: {
    queries: {
      // Trading data goes stale fast — show cache instantly on mount, but
      // *always* fire a background refetch so navigating to a route never
      // shows numbers older than ~RTT. Without this, clicking a navbar
      // link with cache still inside staleTime serves the stale view
      // and no network call happens until staleTime expires.
      refetchOnMount: "always",
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <TooltipProvider delayDuration={250}>
        <RouterProvider router={router} />
        <Toaster richColors closeButton position="top-right" theme="dark" />
        <ReactQueryDevtools initialIsOpen={false} />
      </TooltipProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
