"use client"

import { ColumnDef } from "@tanstack/react-table"

// This type is used to define the shape of our data.
// You can use a Zod schema here if you want.
export type Payment = {
  id: string
  timestamp: string
  source: string
  target: string
  amount: string
  type: string
  fraudulent: boolean
  location: string
}

export const columns: ColumnDef<Payment>[] = [
  {
    accessorKey: "id",
    header: "ID",
  },
  {
    accessorKey: "timestamp",
    header: "Timestamp",
  },
  {
    accessorKey: "source",
    header: "Source",
  },

  {
    accessorKey: "target",
    header: "Target",
  },
  {
    accessorKey: "amount",
    header: "Amount",
  },
  {
    accessorKey: "type",
    header: "Type",
  },
  {
    accessorKey: "fraudulent",
    header: "Fraudulent",
  },
]
