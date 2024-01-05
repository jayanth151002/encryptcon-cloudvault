"use client"
import { MainNav } from "@/components/main-nav";
import {Application, Payment,  columnsApplication, columnsPayment} from "./columns"
import { DataTable } from "@/app/fraud-detection/data-table";
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
    const applications:Application[] = [
      {
        name: "John Doe",
        age: "23",
        debt: "10000",
        dependent: "0",
        maritalStatus: "unmarried",
        accountNumber: "1005426",
        income: "Yes",
        education: "B.Tech",
        status:"Approved",
      },
      ];
      
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
              <h2 className="text-3xl mb-3 font-bold tracking-tight">Credit Card Applications</h2>
              <DataTable columns={columnsApplication} data = {applications}/>
              </div> 
              <div className="pt-6  p-8">
              <h2 className="text-3xl mb-3 font-bold tracking-tight">Fraudulent Transactions</h2>
              <DataTable columns={columnsPayment} data = {payments}/>
              </div> 
            </div>
            
    )
}