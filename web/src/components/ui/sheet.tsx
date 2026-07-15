import * as DialogPrimitive from "@radix-ui/react-dialog"
import { X } from "lucide-react"
import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

export const Sheet = DialogPrimitive.Root
export const SheetTrigger = DialogPrimitive.Trigger
export const SheetClose = DialogPrimitive.Close

export function SheetContent({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/45 backdrop-blur-[2px] data-[state=open]:opacity-100 data-[state=closed]:opacity-0 transition-opacity duration-150" />
      <DialogPrimitive.Content
        className={cn(
          "fixed inset-y-0 right-0 z-50 flex w-full max-w-2xl flex-col border-l border-border bg-background shadow-2xl outline-none transition-[transform,opacity] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] data-[state=closed]:translate-x-full data-[state=open]:translate-x-0",
          className,
        )}
      >
        {children}
        <DialogPrimitive.Close className="absolute right-4 top-4 rounded-md p-1.5 text-muted-foreground outline-none transition-[transform,background-color,color] duration-150 active:scale-[0.97] hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring">
          <X className="size-4" />
          <span className="sr-only">Close</span>
        </DialogPrimitive.Close>
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  )
}

export function SheetHeader({ children }: { children: ReactNode }) {
  return <div className="border-b border-border px-6 py-5">{children}</div>
}

export function SheetTitle({ children }: { children: ReactNode }) {
  return <DialogPrimitive.Title className="pr-10 text-lg font-semibold tracking-tight">{children}</DialogPrimitive.Title>
}

export function SheetDescription({ children }: { children: ReactNode }) {
  return <DialogPrimitive.Description className="mt-1 text-sm text-muted-foreground">{children}</DialogPrimitive.Description>
}
