"use client"
import { MainNav } from "@/components/main-nav";
import {Payment, columns} from "./columns"
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
import { DataTable } from "./data-table";
import { Separator } from "@radix-ui/react-separator";
import { useEffect, useState } from "react";
import {ax as axios} from "@/utils/constants";
export default function Home(){
  const accessToken = localStorage.getItem("accessToken")
  const [payments, setPayments] = useState<Payment[]>([]);
  useEffect(()=>{
    console.log(accessToken)
    axios.get("/transactions",{
      headers:{
        'Authorization' : 'Bearer '+ accessToken 
      }
    }).then(function(response){
      console.log(response.data.data)
      setPayments(response.data.data)
    })
  }, []);
    console.log(payments);
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
              <h2 className="text-3xl mb-3 font-bold tracking-tight">Transactions</h2>
              <DataTable columns={columns} data = {payments}/>
              </div> 
            </div>
            
    )
}