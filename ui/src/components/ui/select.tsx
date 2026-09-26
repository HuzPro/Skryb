import * as SelectPrimitive from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "../../lib/utils";

export interface Option {
  value: string;
  label: string;
}

/** A shadcn-style select over a plain list of options; Radix needs a non-empty value, so "" maps to a sentinel. */
export function Select({
  value,
  options,
  onChange,
  disabled,
  className,
  label,
}: {
  value: string;
  options: Option[];
  onChange: (value: string) => void;
  disabled?: boolean;
  className?: string;
  label?: string;
}) {
  const encode = (raw: string) => (raw === "" ? "\u0000" : raw);
  const decode = (raw: string) => (raw === "\u0000" ? "" : raw);
  return (
    <SelectPrimitive.Root value={encode(value)} onValueChange={(next) => onChange(decode(next))} disabled={disabled ?? false}>
      <SelectPrimitive.Trigger
        aria-label={label}
        className={cn(
          "flex h-8 w-full min-w-0 items-center justify-between gap-2 rounded-md border border-input bg-card px-2.5 text-[13px] shadow-xs outline-none transition-colors duration-150 hover:bg-accent/60 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/25 disabled:opacity-50 [&>span]:truncate",
          className,
        )}
      >
        <SelectPrimitive.Value />
        <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Content
          position="popper"
          sideOffset={4}
          className="z-50 max-h-72 min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-lg border bg-popover p-1 shadow-lg"
        >
          <SelectPrimitive.Viewport>
            {options.map((option) => (
              <SelectPrimitive.Item
                key={option.value}
                value={encode(option.value)}
                className="relative flex h-7 items-center rounded-md pr-7 pl-2 text-[13px] outline-none select-none data-[highlighted]:bg-accent"
              >
                <SelectPrimitive.ItemText>{option.label}</SelectPrimitive.ItemText>
                <SelectPrimitive.ItemIndicator className="absolute right-2">
                  <Check className="size-3.5" />
                </SelectPrimitive.ItemIndicator>
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
}
