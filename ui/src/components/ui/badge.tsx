import { cva, type VariantProps } from "class-variance-authority";
import type * as React from "react";
import { cn } from "../../lib/utils";

const badgeVariants = cva("pill inline-flex shrink-0 items-center gap-1 px-2 py-px text-[11px] font-medium whitespace-nowrap", {
  variants: {
    variant: {
      default: "text-muted-foreground",
      success: "text-success",
      primary: "text-primary",
      warning: "text-warning",
      destructive: "text-destructive",
    },
  },
  defaultVariants: { variant: "default" },
});

export function Badge({ className, variant, ...props }: React.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
