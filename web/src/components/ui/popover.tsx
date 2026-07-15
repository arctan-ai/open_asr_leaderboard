import * as PopoverPrimitive from "@radix-ui/react-popover"
import { cn } from "@/lib/utils"

export const Popover = PopoverPrimitive.Root
export const PopoverTrigger = PopoverPrimitive.Trigger

export function PopoverContent({ className, ...props }: PopoverPrimitive.PopoverContentProps) {
  return (
    <PopoverPrimitive.Portal>
      <PopoverPrimitive.Content
        sideOffset={8}
        align="end"
        className={cn(
          "z-50 w-72 rounded-xl border border-border bg-popover p-3 text-popover-foreground shadow-xl outline-none origin-[var(--radix-popover-content-transform-origin)] transition-[transform,opacity] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] data-[state=closed]:scale-[0.97] data-[state=closed]:opacity-0 data-[state=open]:scale-100 data-[state=open]:opacity-100",
          className,
        )}
        {...props}
      />
    </PopoverPrimitive.Portal>
  )
}
