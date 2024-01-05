"use client"
import { MainNav } from "@/components/main-nav";
import {Application, columns} from "./columns"
import {
    Table,
    TableBody,
    TableCaption,
    TableCell,
    TableFooter,
    TableHead,
    TableHeader,
    TableRow,
  } from "@/components/ui/table"
import { DataTable } from "@/app/fraud-detection/data-table";
import { Separator } from "@radix-ui/react-separator";
   
export default function Home(){

    const invoices:Application[] = [
      {
        name: "Vivek",
        age: "23",
        debt: "10000",
        dependent: "0",
        maritalStatus: "unmarried",
        accountNumber: "1005426",
        income: "Yes",
        education: "B.Tech",
        status:"Approved",
      },
      {
        name: "Hari",
        age: "24",
        debt: "1000",
        dependent: "2",
        maritalStatus: "unmarried",
        accountNumber: "1008426",
        income: "Yes",
        education: "B.Tech",
        status:"Approved",
      },
      {
        name: "Subash",
        age: "19",
        debt: "25000",
        dependent: "0",
        maritalStatus: "unmarried",
        accountNumber: "1058426",
        income: "No",
        education: "High School",
        status:"Approved",
      },
      ]
    return (
        <div className="hidden flex-col md:flex">
            <div className="border-b">
              <div className="flex h-16 items-center px-4">
                <MainNav className="mx-6" />
                <div className="ml-auto flex items-center space-x-4">
                </div>
              </div>
              </div>
              <div className="pt-6  p-8">
              <h2 className="text-3xl mb-3 font-bold tracking-tight"><a href= "/form/loan">Form link</a> for loan application</h2>
              <DataTable columns={columns} data = {invoices}/>
              </div>
              <div className="pt-6  p-8">
              <h2 className="text-3xl mb-3 font-bold tracking-tight">Applications</h2>
              <DataTable columns={columns} data = {invoices}/>
              </div> 
            </div>
            
    )
}