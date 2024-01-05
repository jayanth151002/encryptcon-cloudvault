"use client"
import { useEffect, useState } from "react";
import {ax as axios} from "@/utils/constants";
import { Button } from "@/components/ui/button";
export default function Home(){
    let userId:string = localStorage.getItem("userId")!
    const [image, setImage] = useState<string>('')
    useEffect(()=>{
        console.log(userId)
        axios.get(`/user/qr?id=${userId}`).then(function(response){
            console.log(response);
            setImage(response.data.qr)
        })
    }, []);

    return (
        <div className="flex h-screen">
          <div className="flex-1 flex flex-col overflow-hidden">
            <main className="flex-1 overflow-x-hidden overflow-y-auto ">
              <div className="container mx-auto px-6 py-8">
                <h1 className="text-2xl font-semibold text-gray-700">Scan your QR code</h1>
                <div className="mt-4 p-4 mx-auto bg-white shadow-md rounded-md w-1/3">
                  <p className="text-lg font-medium text-gray-800 mb-4">QR Code </p>
                  <img
                src={"data:image/jpeg;base64," + image} 
                alt="Base64 Image"
                className="w-full h-auto"
              />
                </div>
                <div className = ""><a href="/login/otp"><Button>Next</Button></a></div>
              </div>
            </main>
          </div>
        </div>
      );
}