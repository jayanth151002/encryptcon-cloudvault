"use client"
import { MainNav } from "@/components/main-nav";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import uploadFileToS3 from "@/utils/fileUpload";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { v4 as uuidv4 } from "uuid";

export default function Home(){
    const [file, setFile] = useState<File | null>();
    const router = useRouter();
  const handleFileChange = (e:any) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
  };

  const handleFileUpload = async (e:any) => {
    if(file){
        const fileType = file.name.split(".").pop() ?? "csv";
        const fileName = `${uuidv4()}.${fileType}`;
      try{
        //const s3url = await uploadFileToS3(file, fileName);
        //console.log("");
        router.push('/loan-underwriting/existing')
        // Call Archish's api with this file url and train it
        // after training redirect it to a page similar to already created loan system
      }catch (error) {
        console.error("Error uploading file:", error);
      }
    }
  }
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
              <h2 className="text-3xl mb-3 font-bold tracking-tight">Upload a history of previous loan underwritings</h2>
              <Input type = "file" placeholder = "Upload a file" onChange = {handleFileChange}/>
              <Button className = "my-3" onClick = {handleFileUpload}> Submit</Button>
              </div> 
            </div>
    )
}