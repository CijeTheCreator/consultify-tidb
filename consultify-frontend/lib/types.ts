export interface User {
  id: string
  name: string
  email: string
  language: string
  type: "PATIENT" | "DOCTOR" | "CLERK"
  specialty?: string
}