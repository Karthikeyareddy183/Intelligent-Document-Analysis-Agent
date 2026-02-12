"use client";

import "./globals.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useState } from "react";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 5000,
            retry: 1,
          },
        },
      })
  );

  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen">
        <QueryClientProvider client={queryClient}>
          <TooltipProvider delayDuration={300}>
            {children}
          </TooltipProvider>
          <Toaster
            position="bottom-right"
            richColors
            toastOptions={{
              className: "font-sans",
              style: {
                borderRadius: "0.75rem",
              },
            }}
          />
        </QueryClientProvider>
      </body>
    </html>
  );
}
