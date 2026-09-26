import { cva, type VariantProps } from "class-variance-authority";
import type * as React from "react";
import { cn } from "../../lib/utils";

const badgeVariants = cva("inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-px text-[11px] font-medium whitespace-nowrap", {
  variants: {
    variant: {
      default: "bg-muted text-muted-foreground",
      success: "bg-success/12 text-success",
      primary: "bg-primary/12 text-primary",
      warning: "bg-warning/14 text-warning",
      destructive: "bg-destructive/12 text-destructive",
    },
  },
  defaultVariants: { variant: "default" },
});

export function Badge({ className, variant, ...props }: React.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
