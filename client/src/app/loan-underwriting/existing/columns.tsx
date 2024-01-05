"use client"

import { ColumnDef } from "@tanstack/react-table"

// This type is used to define the shape of our data.
// You can use a Zod schema here if you want.
export type Application = {
  name: string
  age: string
  debt: string
  dependent: string
  maritalStatus: string
  accountNumber: string
  income: "Yes" | "No"
  education:string
  status:string
}

export const columns: ColumnDef<Application>[] = [
  {
    accessorKey: "name",
    header: "Name",
  },
  {
    accessorKey: "age",
    header: "Age",
  },
  {
    accessorKey: "debt",
    header: "Debt",
  },

  {
    accessorKey: "dependent",
    header: "Dependent",
  },

  {
    accessorKey: "maritalStatus",
    header: "Marital Status",
  },

  {
    accessorKey: "accountNumber",
    header: "Account Number",
  },

  {
    accessorKey: "education",
    header: "Education",
  },

  {
    accessorKey:"status",
    header: "Status"
  }
]
