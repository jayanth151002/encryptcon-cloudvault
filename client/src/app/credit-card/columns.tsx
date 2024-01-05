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
export type Payment = {
    id: string
    timestamp: string
    source: string
    target: string
    amount: string
    type: string
    flagged: "Yes" | "No"
    
  }
export const columnsPayment: ColumnDef<Payment>[] = [
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
  
export const columnsApplication: ColumnDef<Application>[] = [
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
