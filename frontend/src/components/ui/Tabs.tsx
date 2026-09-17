import * as React from "react";
import * as TabsPrim from "@radix-ui/react-tabs";
import { cn } from "@/lib/utils";

export const Tabs = TabsPrim.Root;

export function TabsList({ className, ...p }: React.ComponentProps<typeof TabsPrim.List>) {
  return (
    <TabsPrim.List
      {...p}
      className={cn(
        "inline-flex items-center gap-1 border-b border-border",
        "w-full overflow-x-auto hide-native-scrollbar",
        className,
      )}
    />
  );
}

export function TabsTrigger({ className, ...p }: React.ComponentProps<typeof TabsPrim.Trigger>) {
  return (
    <TabsPrim.Trigger
      {...p}
      className={cn(
        "relative h-9 px-3 text-body-sm text-fg-muted hover:text-fg",
        "transition-colors duration-120",
        "data-[state=active]:text-fg",
        "data-[state=active]:after:absolute data-[state=active]:after:bottom-[-1px] data-[state=active]:after:left-0 data-[state=active]:after:right-0 data-[state=active]:after:h-[2px] data-[state=active]:after:bg-accent",
        className,
      )}
    />
  );
}

export const TabsContent = ({ className, ...p }: React.ComponentProps<typeof TabsPrim.Content>) =>
  <TabsPrim.Content {...p} className={cn("pt-4 focus-visible:outline-none", className)} />;
