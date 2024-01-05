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
  const [username, setUsername] = useState<string>('')
  const [password, setPassword]  = useState<string>('')
  const router = useRouter();
  const handleLogin = ()=>{
    console.log(username);
    console.log(password);
    localStorage.setItem("password", password);
     axios.post('/user/verify-password', {
      'username' : username,
      'password': password
     }).then(function (response:any){
      console.log(response)
      if(response.data.success && response.data.qr_scanned)
        {
          localStorage.setItem("userId", response.data.user_id);
          router.push('/login/otp');
        }
        
      else if(response.data.success){
        console.log(response.data.user_id);
        localStorage.setItem("userId", response.data.user_id);
        router.push('/login/qr')
        }
      else alert("Invalid Credentials");
      return;
    });
  }
  return (
    <div className = "min-h-screen flex items-center justify-center" >
      <div className="bg-white p-8 border-solid border-2 rounded-md w-96">
      <h1 className="text-2xl text-slate-800 font-semibold mb-4">Login</h1>
      
        <div className="grid gap-2">
          <div className="grid gap-1">
            <Label className="sr-only" htmlFor="email">
              Email
            </Label>
            <Input
            className = "text-slate-800"
              id="username"
              onChange = {(e)=>setUsername(e.target.value)}
              placeholder="username"
              type="text"
              autoCapitalize="none"
              autoComplete="email"
              autoCorrect="off"
              disabled={isLoading}
            />
            <Input
              className = "text-slate-800"
              onChange = {(e)=>setPassword(e.target.value)}
              id="password"
              placeholder="password"
              type="password"
              autoCapitalize="none"
              autoComplete="password"
              autoCorrect="off"
              disabled={isLoading}
            />
          </div>
          <Button onClick = {handleLogin} disabled={isLoading}>
            Sign In With Username
          </Button>
        </div>
      
      </div>
    </div>
  )
}