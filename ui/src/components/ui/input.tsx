import type * as React from "react";
import { cn } from "../../lib/utils";

const fieldClass =
  "w-full min-w-0 rounded-md border border-input bg-card px-2.5 text-[13px] shadow-xs outline-none transition-colors duration-150 placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/25 disabled:opacity-50 aria-invalid:border-destructive";

export function Input({ className, ...props }: React.ComponentProps<"input">) {
  return <input className={cn(fieldClass, "h-8", className)} spellCheck={false} {...props} />;
}

export function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return <textarea className={cn(fieldClass, "min-h-20 resize-y py-2", className)} {...props} />;
}
