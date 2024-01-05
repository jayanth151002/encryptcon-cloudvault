"use client"
import * as React from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { useState, useEffect } from "react"
import {ax as axios} from "@/utils/constants"
import { useRouter } from "next/navigation"
import { BASE_URL } from "@/utils/constants"
interface loginResponse{
  success:boolean;
  user_id:string;
  qr_scanned:boolean;
}
export default function UserAuthForm() {
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [otp, setOtp] = useState<string>('')
  const password = localStorage.getItem("password")
  const userId = localStorage.getItem("userId")

  const router = useRouter();
  const handleLogin = ()=>{
    console.log(otp);
    console.log(password);
    console.log(userId);
     axios.post('/user/verify-2fa', {
      'id' : userId,
      'code' : otp,
      'password': password
     }).then(function (response:any){
      if(response.data.success){
        localStorage.setItem("accessToken", response.data.token)
        router.push("/dashboard");
      }
    });
  }
  return (
    <div className = "min-h-screen flex items-center justify-center" >
      <div className="bg-white p-8 border-solid border-2 rounded-md w-96">
      <h1 className="text-2xl text-slate-800 font-semibold mb-4">Enter OTP</h1>
      
        <div className="grid gap-2">
          <div className="grid gap-1">
            <Label className="sr-only" htmlFor="email">
              Enter OTP
            </Label>
            <Input
            className = "text-slate-800"
              id="otp"
              onChange = {(e)=>setOtp(e.target.value)}
              placeholder="Enter 6-digit OTP"
              type="text"
              autoCapitalize="none"
              autoComplete="email"
              autoCorrect="off"
              disabled={isLoading}
            />
          </div>
          <Button onClick = {handleLogin} disabled={isLoading}>
            Sign In
          </Button>
        </div>
      
      </div>
    </div>
  )
}