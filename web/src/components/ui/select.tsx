import * as SelectPrimitive from "@radix-ui/react-select"
import { Check, ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"

export const Select = SelectPrimitive.Root
export const SelectValue = SelectPrimitive.Value

export function SelectTrigger({ className, children, ...props }: SelectPrimitive.SelectTriggerProps) {
  return (
    <SelectPrimitive.Trigger
      className={cn(
        "flex h-10 w-full items-center justify-between rounded-lg border border-input bg-background px-3 text-sm shadow-xs outline-none transition-[border-color,box-shadow] duration-150 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/20 disabled:opacity-50",
        className,
      )}
      {...props}
    >
      {children}
      <SelectPrimitive.Icon><ChevronDown className="size-4 text-muted-foreground" /></SelectPrimitive.Icon>
    </SelectPrimitive.Trigger>
  )
}

export function SelectContent({ className, children, ...props }: SelectPrimitive.SelectContentProps) {
  return (
    <SelectPrimitive.Portal>
      <SelectPrimitive.Content
        className={cn(
          "z-50 min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-lg border border-border bg-popover text-popover-foreground shadow-xl origin-[var(--radix-select-content-transform-origin)] data-[state=open]:animate-none",
          className,
        )}
        position="popper"
        sideOffset={6}
        {...props}
      >
        <SelectPrimitive.Viewport className="p-1">{children}</SelectPrimitive.Viewport>
      </SelectPrimitive.Content>
    </SelectPrimitive.Portal>
  )
}

export function SelectItem({ className, children, ...props }: SelectPrimitive.SelectItemProps) {
  return (
    <SelectPrimitive.Item
      className={cn(
        "relative flex cursor-default select-none items-center rounded-md py-2 pr-8 pl-2 text-sm outline-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50",
        className,
      )}
      {...props}
    >
      <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
      <SelectPrimitive.ItemIndicator className="absolute right-2"><Check className="size-4" /></SelectPrimitive.ItemIndicator>
    </SelectPrimitive.Item>
  )
}
