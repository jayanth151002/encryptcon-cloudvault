import axios, { AxiosStatic } from "axios"
export const BASE_URL:string = "https://api-encryptcon.jayanthk.in"

export const ax = axios.create({
    baseURL: BASE_URL
});