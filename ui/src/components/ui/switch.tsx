import * as SwitchPrimitive from "@radix-ui/react-switch";
import type * as React from "react";
import { cn } from "../../lib/utils";

export function Switch({ className, ...props }: React.ComponentProps<typeof SwitchPrimitive.Root>) {
  return (
    <SwitchPrimitive.Root
      className={cn(
        "peer inline-flex h-5 w-9 shrink-0 items-center rounded-[var(--r-switch)] border border-[var(--track-line)] bg-[var(--track-off)] shadow-xs outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring/50 disabled:opacity-50 data-[state=checked]:border-transparent data-[state=checked]:bg-primary",
        className,
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb className="pointer-events-none block size-4 translate-x-0.5 scale-[var(--thumb-off-scale)] rounded-[calc(var(--r-switch)-2px)] bg-[var(--thumb)] shadow-[var(--thumb-shadow)] spring data-[state=checked]:translate-x-[18px] data-[state=checked]:scale-100 data-[state=checked]:bg-[var(--thumb-on)]" />
    </SwitchPrimitive.Root>
  );
}
